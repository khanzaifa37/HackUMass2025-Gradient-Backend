from ai_utils import setup_auth
from prompts import SAFETY_SETTINGS
import requests
import tempfile
import shutil
import os
import json
import google.generativeai as genai
from google.generativeai import types

def setupSupabse(supabase_url: str, supabase_key: str):
    SUPABASE_URL = supabase_url or os.environ.get("NEXT_PUBLIC_SUPABASE_URL") or os.environ.get("SUPABASE_URL")
    SUPABASE_KEY = supabase_key or os.environ.get("SUPABASE_SERVICE_ROLE_KEY") or os.environ.get("NEXT_PUBLIC_SUPABASE_ANON_KEY")

    if not SUPABASE_URL or not SUPABASE_KEY:
        raise Exception("Supabase URL or key not provided via args or environment variables")

def callGeminiInsights(formatted_results):
    model = genai.GenerativeModel(model_name="gemini-2.5-flash")

    prompt = f"""
    You are an expert academic grader and pedagogy analyst. Your role is to review a provided assignment PDF
    (containing the problem set and correct solutions) along with a batch of student submissions (each containing
    the student\’s answers). You specialize in advanced academic subjects such as Machine Learning, Advanced NLP,
    Game Theory, and other graduate-level domains. Maintain a professional, precise, and objective tone.

    Results from grading: {formatted_results}

    TASKS:
    1. Parse the Assignment PDF:
    - Identify each question and its expected correct reasoning steps.
    - Understand the correct final answers and conceptual foundations.

    2. Analyze Student Submissions:
    - Evaluate correctness of each answer for each student.
    - Accept differences in phrasing if conceptual understanding is accurate.
    - Focus on mathematical, procedural, and conceptual accuracy rather than formatting.

    3. Identify Misunderstood Content:
    - Calculate the incorrect answer rate per question across all students.
    - Identify questions/topics with the highest error frequencies.
    - Determine underlying conceptual gaps or common reasoning mistakes.

    4. Your final output MUST be a single valid JSON object with the following structure:
    {{
    "overview": {{
        "overall_accuracy_rate": <number>,
        "total_submissions": <number>,
        "average_score": <number>
    }},
    "most_missed_questions": [
        {{
        "question_number": "1.a",
        "incorrect_rate": <number>,
        "related_topic": <string>,
        "summary_of_difficulty": <string>
        }}
    ],
    "misunderstood_topics": [
        {{
        "topic": <string>,
        "reason_for_difficulty": <string>
        }}
    ],
    "recommended_review_topics": [
        {{
        "topic": <string>,
        "instructional_impact_reason": <string>
        }}
    ]
    }}
    ADDITIONAL REQUIREMENTS:
    - The output MUST be valid JSON. Do not include commentary outside the JSON.
    - Do not include student names, IDs, or any personal identifiers.
    - If data is missing, use null instead of leaving fields out.
    - Assume students answer in good faith; focus on actionable educational insight.
    """
    insight_prompt = prompt

    try:
        response = model.generate_content(
            insight_prompt,
            generation_config=types.GenerationConfig(
                temperature=0.1,
                max_output_tokens=2000000
            ),
            safety_settings=SAFETY_SETTINGS
        )
        
        # Check if response was blocked
        if not response.candidates:
            return {
                "error": "Response blocked by safety filters",
                "finish_reason": "SAFETY",
                "detail": "No candidates returned"
            }
        
        # Check finish reason
        candidate = response.candidates[0]
        if candidate.finish_reason != 1:  # 1 = STOP (normal completion)
            finish_reasons = {
                0: "FINISH_REASON_UNSPECIFIED",
                1: "STOP",
                2: "SAFETY",
                3: "RECITATION",
                4: "OTHER"
            }
            return {
                "error": "Response not completed normally",
                "finish_reason": finish_reasons.get(candidate.finish_reason, "UNKNOWN"),
                "safety_ratings": [
                    {
                        "category": rating.category,
                        "probability": rating.probability
                    } for rating in candidate.safety_ratings
                ]
            }
        
        return response.text
        
    except Exception as e:
        return {
            "error": "Exception during generation",
            "detail": str(e)
        }

def generate_insights_from_results(assignment_id: str, supabase_url:str, supabase_key:str):
    setup_auth()
    setupSupabse(url, key)
    tmpdir = tempfile.mkdtemp(prefix="results_")
    results_url = f"{supabase_url.rstrip('/')}/rest/v1/results"
    params = {"select": "*", "assignment_id": f"eq.{assignment_id}"}
    headers = {
        "apikey": supabase_key,
        "Authorization": f"Bearer {supabase_key}",
        "Accept": "application/json"
    }
    resp = requests.get(results_url, params=params, headers=headers, timeout=30)
    if resp.status_code != 200:
        raise Exception(f"Failed to fetch results from Supabase: {resp.status_code} {resp.text}")
    results = resp.json()
    print("===========Results from results table============")
    print(results)
    if not results:
        return {"status": "no_results", "message": "No graded submissions found for this assignment."}

    # Extract all grading structures
    grading_data = []
    for entry in results:
        gd = entry.get("result_json")
        if gd:
            grading_data.append(gd)
    print("===========Grading data retrieved:============")
    print(grading_data)
    if not grading_data:
        return {"status": "no_grading_data", "message": "No grading data available for analysis."}
    
    formatted_results = json.dumps(grading_data, indent=2)
    print("\n🔍 Generating insight report...")
    try:
        insight_report = callGeminiInsights(formatted_results)  # Replace with your model call wrapper
    except Exception as e:
        raise Exception(f"Insight LLM generation failed: {e}")
    # Optional: store insights back to Supabase (e.g., in `assignment_insights` table)
    try:
        insights_upload_url = f"{supabase_url.rstrip('/')}/rest/v1/insights"
        payload = {
            "assignment_id": assignment_id,
            "insights_json": insight_report
        }
        upload_resp = requests.post(
            insights_upload_url,
            headers={
                "apikey": supabase_key,
                "Authorization": f"Bearer {supabase_key}",
                "Content-Type": "application/json"
            },
            data=json.dumps(payload)
        )
        print("========Insights Report==========")
        print(insight_report)

        if upload_resp.status_code not in (200, 201):
            print(f"Warning: Could not store insights: {upload_resp.status_code} {upload_resp.text}")

    except Exception as e:
        print(f"Warning: Failed to upload insights: {e}")

    try:
        shutil.rmtree(tmpdir)
    except Exception:
        pass

    return {
        "status": "success",
        "assignment_id": assignment_id,
        "insights_report": insight_report
    }

if __name__ == "__main__":
    print("🚀 Starting insights testing...")
    url = os.environ.get("NEXT_PUBLIC_SUPABASE_URL") or os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY") or os.environ.get("NEXT_PUBLIC_SUPABASE_ANON_KEY")
    generate_insights_from_results("0a54f3a4-0e7e-47d1-9054-2058a9b8ccd5", url, key)

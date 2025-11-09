import os
import time
import google.generativeai as genai
from google.generativeai import types
import uuid
import sys
import tempfile
import shutil
import requests
import json
from datetime import datetime, timezone
import time
import random
from typing import List, Dict, Any, Optional
from urllib.parse import urlparse, unquote,urlunparse




def setup_auth():
    """Sets up authentication for the Gemini API by checking for an env var."""
    try:
        api_key = os.environ["GEMINI_API_KEY"]
        genai.configure(api_key=api_key)
        print("Authentication configured using GOOGLE_API_KEY.")
    except KeyError:
        print("Error: GOOGLE_API_KEY environment variable not set.")
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred during authentication setup: {e}")
        sys.exit(1)


def grade_student_answer(rubric_text: str, question_text: str, student_answer: str, model_name: str = "gemini-2.5-flash"):
    model = genai.GenerativeModel(model_name=model_name)
    grading_prompt = f"""
    You are an expert teacher grading a student's submission.
    
    Questions:
    {question_text}

    Rubric (each question's grading criteria):
    {rubric_text}

    Student's Answers:
    {student_answer}

    ---
    TASK:
    1. Identify each question number (like 1.a, 1.b, etc.).
    2. For each question, use the rubric to decide a numeric score.
    3. Provide a short reason for why that score fits the rubric.
    4. Suggest how the student can improve.
    5. Be extremely strict in the marking. If a question mentions do not add anything in your report for this question, give the student full marks.

    Only use numeric scores listed in the rubric. Do not invent new scales.

    OUTPUT FORMAT:
    {{
      "results": [
        {{
          "question": "1.a",
          "score": <number>,
          "reason": "<reason based on rubric>",
          "improvement": "<how to improve>"
        }},
        ...
      ],
      "overall_feedback": "<overall comment summarizing performance>"
      "total_score":"<The total score achieved by the student>"
    }}
    """
    
    # Add safety settings to allow educational content
    safety_settings = [
        {
            "category": "HARM_CATEGORY_HARASSMENT",
            "threshold": "BLOCK_NONE"
        },
        {
            "category": "HARM_CATEGORY_HATE_SPEECH",
            "threshold": "BLOCK_NONE"
        },
        {
            "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
            "threshold": "BLOCK_NONE"
        },
        {
            "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
            "threshold": "BLOCK_NONE"
        }
    ]
    
    try:
        response = model.generate_content(
            grading_prompt,
            generation_config=types.GenerationConfig(
                temperature=0.1,
                max_output_tokens=2000000
            ),
            safety_settings=safety_settings
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

def transcribe_pdf_from_path(pdf_path: str, system_prompt: str, model_name: str = "gemini-2.5-flash"):
    try:
        model = genai.GenerativeModel(
            model_name=model_name,
            system_instruction=system_prompt
        )
    except Exception as e:
        return f"Error: Could not instantiate model {model_name}."

    pdf_file = None
    try:
        pdf_file = genai.upload_file(
            path=pdf_path,
            display_name=os.path.basename(pdf_path)
        )

        while pdf_file.state.name == "PROCESSING":
            time.sleep(10)
            pdf_file = genai.get_file(name=pdf_file.name)

        if pdf_file.state.name != "ACTIVE":
            raise Exception(f"File processing failed. Final state: {pdf_file.state.name}")

        response = model.generate_content(
            [pdf_file, "Please transcribe this document following all instructions."],
            generation_config=types.GenerationConfig(
                max_output_tokens=15000,
                temperature=0.0
            )
        )
        text_output = response.text

    except Exception as e:
        text_output = f"Error: {e}"
    finally:
        if pdf_file:
            try:
                genai.delete_file(name=pdf_file.name)
            except Exception:
                pass

    return text_output
def construct_full_storage_url(file_path: str, supabase_url: str, bucket_name: str) -> str:
    """Construct full Supabase storage URL from various input formats."""
    if file_path.startswith("http://") or file_path.startswith("https://"):
        return file_path
    
    file_path = file_path.lstrip("/")
    
    if not file_path.startswith(f"{bucket_name}/"):
        file_path = f"{bucket_name}/{file_path}"
    
    return f"{supabase_url.rstrip('/')}/storage/v1/object/public/{file_path}"


def get_signed_url(file_path: str, supabase_url: str, supabase_key: str, 
                   bucket_name: str, expires_in: int = 604800) -> str:
    """Get signed URL from Supabase storage."""
    full_url = construct_full_storage_url(file_path, supabase_url, bucket_name)
    
    if "token=" in full_url:
        return full_url
    
    parsed = urlparse(full_url)
    path = parsed.path
    
    marker = "/storage/v1/object/public/"
    idx = path.find(marker)
    if idx == -1:
        raise ValueError(f"URL malformed: {full_url}")
    
    after_marker = path[idx + len(marker):]
    parts = after_marker.split('/', 1)
    if len(parts) != 2:
        raise ValueError(f"Cannot parse bucket/path: {full_url}")
    
    bucket, file_path_clean = parts
    
    sign_url = f"{supabase_url.rstrip('/')}/storage/v1/object/sign/{bucket}/{file_path_clean}"
    
    headers = {
        "apikey": supabase_key,
        "Authorization": f"Bearer {supabase_key}",
        "Content-Type": "application/json"
    }
    
    payload = {"expiresIn": expires_in}
    
    print(f"   Requesting signed URL from: {sign_url}")
    resp = requests.post(sign_url, json=payload, headers=headers, timeout=10)
    
    if resp.status_code != 200:
        raise Exception(f"Sign request failed: {resp.status_code} - {resp.text}")
    
    data = resp.json()
    signed_path = data.get("signedURL").split('?token')[-1]
    print(data)
    if not signed_path:
        raise Exception(f"No signedURL in response: {data}")
    
    return sign_url+"?token"+signed_path

def grade_submissions_for_assignment(assignment_id: str, assignment_idea: Optional[str], supabase_url: Optional[str] = None, supabase_key: Optional[str] = None) -> Dict[str, Any]:
    """Fetch submissions for an assignment from Supabase, transcribe and grade each one.

    Parameters:
      - assignment_id: ID used in the `submissions` table to filter rows.
      - assignment_idea: rubric or assignment description text passed to grader.
      - supabase_url/supabase_key: optional overrides for environment variables.

    Returns a dict: {"count": n, "results": [...] }
    """
    setup_auth()

    SUPABASE_URL = supabase_url or os.environ.get("NEXT_PUBLIC_SUPABASE_URL") or os.environ.get("SUPABASE_URL")
    SUPABASE_KEY = supabase_key or os.environ.get("SUPABASE_SERVICE_ROLE_KEY") or os.environ.get("NEXT_PUBLIC_SUPABASE_ANON_KEY")

    if not SUPABASE_URL or not SUPABASE_KEY:
        raise Exception("Supabase URL or key not provided via args or environment variables")

    tmpdir = tempfile.mkdtemp(prefix="submissions_")
    PROMPT_ANSWERSCRIPT = (
        "You are an expert transcriptionist specializing in handwritten documents."
        "Transcribe the attached PDF, which contains handwritten questions."
        "Your task is to produce a clean, plain-text version of the content."
        "Follow these rules precisely:"
        "1. Preserve the question format."
        "2. Start each question with the prefix 'Question:' on a new line."
        "3. For any handwritten math, transcribe it into clear, readable LaTeX format (e.g., $E = mc^2$, $\\frac{a}{b}$)."
    )
    questions_url = f"{SUPABASE_URL.rstrip('/')}/rest/v1/assignments"
    params = {"select": "*", "id": f"eq.{assignment_id}"}
    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Accept": "application/json"
    }
    resp = requests.get(questions_url, params=params, headers=headers, timeout=30)
    if resp.status_code != 200:
        raise Exception(f"Failed to fetch questions from Supabase: {resp.status_code} {resp.text}")
    questions = resp.json()
    question_txt = ""
    rubric_txt = ""
    for question in questions:
        try:
            file_url = question.get("file_url")
            rubric_url = question.get("rubric_path")
            signed_url = get_signed_url(file_url, SUPABASE_URL, SUPABASE_KEY, "assignments")
            print(f"   Signed URL: {signed_url}")
            
            signed_resp = requests.get(signed_url, timeout=10)
            print(f"   Signed response status: {signed_resp.status_code}")
            
            if signed_resp.status_code == 200:
                print(f"   ✅ Signed download successful!")
                local_name = os.path.join(tmpdir, f"{uuid.uuid4()}_{os.path.basename(file_url)}")
                with open(local_name, "wb") as f:
                    f.write(signed_resp.content)
                
                question_txt = transcribe_pdf_from_path(local_name, PROMPT_ANSWERSCRIPT)
            else:
                print(f"   ❌ Signed download failed: {signed_resp.text[:200]}")
            signed_url = get_signed_url(rubric_url, SUPABASE_URL, SUPABASE_KEY, "rubric")
            print(f"   Signed URL: {signed_url}")
            
            signed_resp = requests.get(signed_url, timeout=10)
            print(f"   Signed response status: {signed_resp.status_code}")
            
            if signed_resp.status_code == 200:
                print(f"   ✅ Signed download successful!")
                local_name = os.path.join(tmpdir, f"{uuid.uuid4()}_{os.path.basename(rubric_url)}")
                with open(local_name, "wb") as f:
                    f.write(signed_resp.content)
                
                rubric_txt = transcribe_pdf_from_path(local_name, PROMPT_ANSWERSCRIPT)
            else:
                print(f"   ❌ Signed download failed: {signed_resp.text[:200]}")
        except Exception as e:
            print(f"   ❌ Signed URL failed: {e}")
    rest_url = f"{SUPABASE_URL.rstrip('/')}/rest/v1/submissions"
    params = {"select": "*", "assignment_id": f"eq.{assignment_id}"}
    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Accept": "application/json"
    }
    resp = requests.get(rest_url, params=params, headers=headers, timeout=30)
    if resp.status_code != 200:
        raise Exception(f"Failed to fetch submissions from Supabase: {resp.status_code} {resp.text}")

    submissions = resp.json()

    results: List[Dict[str, Any]] = []
    PROMPT_ANSWERSCRIPT = (
        "You are an expert transcriptionist specializing in handwritten documents."
        "Transcribe the attached PDF, which contains handwritten answers."
        "Your task is to produce a clean, plain-text version of the content."
        "Follow these rules precisely:"
        "1. Preserve the answer format."
        "2. Start each answer with the prefix 'Answer:' on a new line."
        "3. For any handwritten math, transcribe it into clear, readable LaTeX format (e.g., $E = mc^2$, $\\frac{a}{b}$)."
    )
    try:
        for sub in submissions:
            try:
                user_id = sub.get("user_id")
                file_url = sub.get("file_url")
                submission_id = sub.get("id")
                
                print(f"\n📄 Processing submission:")
                print(f"   Submission ID: {submission_id}")
                print(f"   User ID: {user_id}")
                print(f"   Raw file_url: '{file_url}'")
                
                if not file_url:
                    results.append({
                        "submission_id": submission_id,
                        "user_id": user_id,
                        "status": "skipped",
                        "reason": "no public file_url present"
                    })
                    continue

                # Approach 2: Signed URL
                print(f"\n🔄 Attempt 2: Signed URL")
                try:
                    signed_url = get_signed_url(file_url, SUPABASE_URL, SUPABASE_KEY, "submissions")
                    print(f"   Signed URL: {signed_url}")
                    
                    signed_resp = requests.get(signed_url, timeout=10)
                    print(f"   Signed response status: {signed_resp.status_code}")
                    
                    if signed_resp.status_code == 200:
                        print(f"   ✅ Signed download successful!")
                        local_name = os.path.join(tmpdir, f"{uuid.uuid4()}_{os.path.basename(file_url)}")
                        with open(local_name, "wb") as f:
                            f.write(signed_resp.content)
                        
                        student_text = transcribe_pdf_from_path(local_name, PROMPT_ANSWERSCRIPT)
                        grading = grade_student_answer(rubric_text=rubric_txt, question_text=question_txt, student_answer=student_text)
                        print(grading)
                        results.append({
                            "submission_id": submission_id,
                            "user_id": user_id,
                            "status": "graded",
                            "grading": grading
                        })
                        upload_results(SUPABASE_URL, SUPABASE_KEY, submission_id, user_id, "graded", grading, assignment_id)
                        continue
                    else:
                        print(f"   ❌ Signed download failed: {signed_resp.text[:200]}")
                except Exception as e:
                    print(f"   ❌ Signed URL failed: {e}")

                # If both failed
                results.append({
                    "submission_id": submission_id,
                    "user_id": user_id,
                    "status": "download_failed",
                    "detail": "Both direct and signed URL approaches failed"
                })
                print("Upload Results starting")
                upload_results(SUPABASE_URL, SUPABASE_KEY, submission_id, user_id, "failed", grading, assignment_id)
            except Exception as e:
                print(f"❌ Error processing submission: {e}")
                results.append({
                    "submission_id": sub.get("id"),
                    "status": "error",
                    "detail": str(e)
                })
    finally:
        try:
            shutil.rmtree(tmpdir)
        except Exception:
            pass
        
        
    return {"count": len(results), "results": results}

def generate_unique_bigint():
    timestamp_ms = int(time.time() * 1000)  # Current time in milliseconds
    random_part = random.randint(0, 99999)  # Add a random component
    # Combine timestamp and random part. Ensure it fits within BIGINT limits.
    # PostgreSQL BIGINT range: -9,223,372,036,854,775,808 to 9,223,372,036,854,775,807
    unique_id = (timestamp_ms * 100000) + random_part 
    return unique_id

def update_submission_status(
    supabase_url: str,
    supabase_key: str,
    submission_id: str,
    new_status: str
) -> bool:
    if not submission_id:
        print("Update Status Error: submission_id is missing. Skipping update.")
        return False

    print(f"Updating status for submission_id '{submission_id}' to '{new_status}'...")
    
    # Use the 'eq' filter to target the specific row
    rest_url = f"{supabase_url.rstrip('/')}/rest/v1/submissions?id=eq.{submission_id}"
    headers = {
        "apikey": supabase_key,
        "Authorization": f"Bearer {supabase_key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Prefer": "return=minimal" # Don't return the updated object
    }
    payload = {
        "status": new_status
    }

    try:
        response = requests.patch(rest_url, headers=headers, json=payload, timeout=30)
        
        # Check for success (2xx)
        response.raise_for_status()
        
        print(f"🚦🚦🚦🚦🚦🚦Successfully updated status for submission '{submission_id}'.")
        return True
    
    except requests.exceptions.HTTPError as e:
        print(f"Error updating submission status: {e.response.status_code}")
        print(f"Response body: {e.response.text}")
    except requests.exceptions.RequestException as e:
        print(f"A network error occurred: {e}")
    
    return False

def upload_results(SUPABASE_URL: str,
    SUPABASE_KEY: str,
    submission_id: str,
    user_id: str,
    processing_status: str,
    raw_results_text: str,
    assignment_id: str
):
    """
    Parses a raw result text, calculates the total score, and uploads the
    complete record to the Supabase 'submissions' table.

    Args:
        SUPABASE_URL: The base URL of your Supabase project.
        SUPABASE_KEY: Your Supabase anon or service role key.
        submission_id: The UUID for the submission.
        user_id: The UUID for the user.
        processing_status: The current status (e.g., "completed").
        raw_results_text: A string containing the JSON results from the model.
    """
    
    print("Starting results upload...")

    try:
        

        # 4. Get the current time in UTC ISO format (for 'created_at')
        created_at_time = datetime.now(timezone.utc).isoformat()

        # 5. Define the Supabase REST endpoint and headers
        # We are inserting into the 'submissions' table
        rest_url = f"{SUPABASE_URL.rstrip('/')}/rest/v1/results"
        headers = {
            "apikey": SUPABASE_KEY,
            "Authorization": f"Bearer {SUPABASE_KEY}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Prefer": "return=minimal" # Asks Supabase to just return 201 on success
        }

        # 6. Build the payload matching the table structure from your image
        # We send a list containing one object to insert a single row
        # --- NEW LOGIC BRANCH ---
        if processing_status == "failed":
            print("Processing a 'failed' submission. Omitting results.")
            
            # Set optional fields to None (for NULL in Supabase)
            payload = [{
                "result_id": generate_unique_bigint(),
                "created_at": created_at_time,
                "submission_id": submission_id,
                "user_id": user_id,
                "processing_status": processing_status,
                "overall_feedback": None,
                "result_json": None,
                "overall_score": None,
                "assignment_id": assignment_id
            }]
        else:
            if not raw_results_text:
                raise ValueError("Processing status is not 'failed' but raw_results_text is empty.")
            
            # --- NEW: Add text processing to clean the raw string ---
            # Remove leading/trailing whitespace
            cleaned_text = raw_results_text.strip()
            
            cleaned_text = cleaned_text.lstrip()          # remove leading spaces/newlines

            if cleaned_text.startswith("```json"):
                cleaned_text = cleaned_text[len("```json"):].strip()
            elif cleaned_text.startswith("```"):
                cleaned_text = cleaned_text[len("```"):].strip()

            # Remove trailing ``` if present
            if cleaned_text.endswith("```"):
                cleaned_text = cleaned_text[:-3].strip()
            
            # Slice the string from the first '{' to the last '}' to get the JSON
            # results_data = cleaned_text[first_brace : last_brace + 1]
            results_data = json.loads(cleaned_text)
            
            # # 1. Parse the raw text blob into a Python dictionary
            # results_data = json.loads(raw_results_text)

            # 2. Extract the relevant fields from the parsed data
            result_json_list = results_data.get("results", [])
            overall_feedback = results_data.get("overall_feedback", "")


            # 3. Calculate the overall_score by summing scores from the results list
            overall_score = 0
            for item in result_json_list:
                # Use .get() for safety, defaulting to 0 if 'score' is missing
                overall_score += item.get("score", 0)
            result_id = generate_unique_bigint()
            print(f"Calculated overall_score: {overall_score}")
            print(f"Submission Id:{submission_id}")
            print(f"ResultID:{result_id}")
            print(f"User Id:{user_id}")

            payload = [
                {
                    "created_at": created_at_time,
                    "result_id":  result_id,
                    "submission_id": submission_id,
                    "user_id": user_id,
                    "processing_status": processing_status,
                    "overall_feedback": overall_feedback,
                    "result_json": result_json_list,  # 'requests' will serialize this to JSON
                    "overall_score": overall_score,
                    "assignment_id": assignment_id
                }
            ]

            update_success = update_submission_status(
                SUPABASE_URL,
                SUPABASE_KEY,
                submission_id,
                "graded" # "graded" or "failed"
            )
        print(f"Sending data to Supabase at: {rest_url}")
        response = requests.post(rest_url, headers=headers, data=json.dumps(payload), timeout=30)
        response.raise_for_status()  # Raises an HTTPError for bad responses (4xx or 5xx)
        print(f"Successfully uploaded submission! Status Code: {response.status_code}")
        return True

    except json.JSONDecodeError:
        print(f"Error: Failed to decode 'raw_results_text'. Check if it's valid JSON.")
        print(f"Raw text was: {raw_results_text}")
        return False
    except requests.exceptions.HTTPError as e:
        print(f"Error uploading to Supabase: {e.response.status_code}")
        print(f"Response body: {e.response.text}")
        return False
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return False
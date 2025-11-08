import os
import time
import google.generativeai as genai
from google.generativeai import types
import uuid
import sys

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


def grade_student_answer(rubric_text: str, student_answer: str, model_name: str = "gemini-1.5-pro-latest"):
    model = genai.GenerativeModel(model_name=model_name)
    grading_prompt = f"""
    You are an expert teacher grading a student's submission.

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
    }}
    """
    response = model.generate_content(
        grading_prompt,
        generation_config=types.GenerationConfig(
            temperature=0.1,
            max_output_tokens=1000
        )
    )
    return response.text


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

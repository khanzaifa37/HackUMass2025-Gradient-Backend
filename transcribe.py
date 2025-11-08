# Package Install Instructions:
# pip install google-generativeai
#
# API key set in environment.

import os
import sys
import time
import google.generativeai as genai
from google.generativeai import types
import uuid

def setup_auth():
    """Sets up authentication for the Gemini API by checking for an env var."""
    try:
        # Check if GOOGLE_API_KEY is set in the environment
        api_key = os.environ["GEMINI_API_KEY"]
        genai.configure(api_key=api_key)
        print("Authentication configured using GOOGLE_API_KEY.")
    except KeyError:
        print("Error: GOOGLE_API_KEY environment variable not set.")
        print("Please get an API key from Google AI Studio and set it as:")
        print("export GOOGLE_API_KEY='your-api-key-here'")
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred during authentication setup: {e}")
        sys.exit(1)
def grade_student_answer(rubric_text: str, student_answer: str, model_name: str = "gemini-1.5-pro-latest"):
    """
    Grades a student's multi-question answer based on a multi-question rubric.
    Returns structured JSON with per-question scoring and improvement suggestions.
    """

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

def transcribe_pdf_from_path(pdf_path: str, model_name: str = "gemini-2.5-flash"):
    """
    Transcribes a PDF file using the Gemini API, handling file upload and cleanup.
    
    Args:
        pdf_path: The local path to the PDF file.
        model_name: The Gemini model to use (e.g., 'gemini-1.5-flash-latest').

    Returns:
        The transcribed text as a string, or an error message.
    """
    
    # Define the system prompt first.
    # We will pass this to the model to guide its transcription.
    system_prompt = (
        "You are an expert transcriptionist specializing in handwritten documents."
        "Transcribe the attached PDF, which contains handwritten questions and answers."
        "Your task is to produce a clean, plain-text version of the content."
        "Follow these rules precisely:"
        "1. Preserve the question and answer (Q&A) format."
        "2. Start each question with the prefix 'Question:' on a new line."
        "3. Start each answer with the prefix 'Answer:' on a new line."
        "4. For any handwritten math, transcribe it into clear, readable LaTeX format (e.g., $E = mc^2$, $\\frac{a}{b}$)."
    )
    
    # 1. Instantiate the model.
    # Use a model that supports file inputs, like 1.5 Flash or Pro.
    # We pass the system_instruction here for broad compatibility.
    try:
        model = genai.GenerativeModel(
            model_name=model_name,
            system_instruction=system_prompt
        )
    except Exception as e:
        print(f"Error creating model: {e}")
        return f"Error: Could not instantiate model {model_name}."

    pdf_file = None  # Initialize to None for cleanup logic
    try:
        # 2. Upload the file to the Gemini API's temporary storage.
        print(f"Uploading file: {pdf_path}...")
        # genai.upload_file returns a File object.
        pdf_file = genai.upload_file(
            path=pdf_path,
            display_name=os.path.basename(pdf_path)
        )

        print(f"File uploaded: {pdf_file.name} (URI: {pdf_file.uri})")

        # 3. CRITICAL: Wait for the file to be processed.
        # You cannot use the file in a prompt until its state is 'ACTIVE'.
        print(f"Current file state: {pdf_file.state.name}")
        while pdf_file.state.name == "PROCESSING":
            print("File is processing, waiting 10 seconds...")
            time.sleep(10)
            # Fetch the file's latest metadata
            pdf_file = genai.get_file(name=pdf_file.name)
            print(f"Current file state: {pdf_file.state.name}")

        if pdf_file.state.name != "ACTIVE":
            raise Exception(f"File processing failed. Final state: {pdf_file.state.name}")

        print("File is ACTIVE. Sending transcription request...")

        # 4. Build the user prompt.
        # The system_prompt is now part of the model.
        user_prompt = "Please transcribe this document following all instructions."

        # 5. Make the generate_content call
        # We pass the file and the user prompt.
        response = model.generate_content(
            [pdf_file, user_prompt],
            generation_config=types.GenerationConfig(
                max_output_tokens=15000,  # Generous limit for a doc
                temperature=0.0  # Low temp for deterministic transcription
            )
        )

        # 6. Get the text from the response
        text_output = response.text

    except Exception as e:
        print(f"An error occurred during file upload and transcribe: {e}")
        # Access more detailed error info if available
        if hasattr(e, 'response'):
            print(f"API Response Error: {e.response}")
        text_output = f"Error: {e}"
    finally:
        # 7. IMPORTANT: Clean up the uploaded file
        # Files persist for 48 hours if not deleted.
        if pdf_file:
            try:
                print(f"Cleaning up file: {pdf_file.name}...")
                genai.delete_file(name=pdf_file.name)
                print("File deleted successfully.")
            except Exception as e:
                print(f"Error deleting file {pdf_file.name}: {e}")
                print("You may need to delete it manually from Google AI Studio.")

    return text_output

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python gemini_pdf_transcribe_fixed.py <file.pdf>")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    
    if not os.path.exists(pdf_path):
        print(f"Error: File not found at {pdf_path}")
        sys.exit(1)

    # Setup auth first
    setup_auth()

    print(f"Starting transcription for: {pdf_path}...")
    result = transcribe_pdf_from_path(pdf_path)
    
    print("\n" + "="*30)
    print("=== TRANSCRIPTION START ===")
    print("="*30 + "\n")
    print(result)
    print("\n" + "="*30)
    print("=== TRANSCRIPTION END ===")
    print("="*30)

    if not result.startswith("Error:"):
        try:
            # Create a unique filename based on your request
            unique_id = str(uuid.uuid4())
            base_name = os.path.basename(pdf_path)
            name_without_ext = os.path.splitext(base_name)[0]
            
            # Format: [UUID]_[input_filename_no_ext]_output.txt
            output_filename = f"{unique_id}_{name_without_ext}_output.txt"
            
            with open(output_filename, "w", encoding="utf-8") as f:
                f.write(result)
            
            print(f"\nSuccessfully saved transcription to: {output_filename}")
        
        except Exception as e:
            print(f"\nError saving transcription to file: {e}")
    else:
        print("\nTranscription failed. Output file was not created.")
    # -------------------------------
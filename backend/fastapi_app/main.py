import os
import uuid
import tempfile
import shutil
from typing import List, Dict, Any

import requests
from fastapi import FastAPI, UploadFile, File, HTTPException, Body
from fastapi.responses import JSONResponse
from .ai_utils import (
    setup_auth,
    transcribe_pdf_from_path,
    grade_student_answer,
    grade_submissions_for_assignment
)

app = FastAPI(title="AI Graded Assignments API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Ensure output folder exists
OUTPUT_DIR = "output_files"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ------------------------------
# Transcribe Answer Script Endpoint
# ------------------------------
@app.post("/transcribe/answer")
async def transcribe_answer(file: UploadFile = File(...)):
    try:
        temp_pdf_path = f"temp_{file.filename}"
        with open(temp_pdf_path, "wb") as f:
            f.write(await file.read())

        setup_auth()

        PROMPT_ANSWERSCRIPT = (
            "You are an expert transcriptionist specializing in handwritten documents."
            "Transcribe the attached PDF, which contains handwritten questions and answers."
            "Your task is to produce a clean, plain-text version of the content."
            "Follow these rules precisely:"
            "1. Preserve the question and answer (Q&A) format."
            "2. Start each question with the prefix 'Question:' on a new line."
            "3. Start each answer with the prefix 'Answer:' on a new line."
            "4. For any handwritten math, transcribe it into clear, readable LaTeX format (e.g., $E = mc^2$, $\\frac{a}{b}$)."
        )

        result_text = transcribe_pdf_from_path(temp_pdf_path, PROMPT_ANSWERSCRIPT)

        output_filename = f"{uuid.uuid4()}_{os.path.splitext(file.filename)[0]}_answer_output.txt"
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(result_text)

        os.remove(temp_pdf_path)

        return JSONResponse(content={
            "filename": output_filename,
            "content": result_text
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ------------------------------
# Transcribe Rubric Endpoint
# ------------------------------
@app.post("/transcribe/rubric")
async def transcribe_rubric(file: UploadFile = File(...)):
    try:
        temp_pdf_path = f"temp_{file.filename}"
        with open(temp_pdf_path, "wb") as f:
            f.write(await file.read())

        setup_auth()

        PROMPT_RUBRIC = (
            "You are an AI assistant specializing in educational assessment."
            "Analyze the attached PDF, which appears to be a scoring rubric or grading guide."
            "Your task is to extract and transcribe this rubric into a clean, plain-text format."
            "Preserve all scoring criteria, sub-criteria, and their associated point values."
            "Structure the output logically, clearly linking criteria to their points."
        )

        result_text = transcribe_pdf_from_path(temp_pdf_path, PROMPT_RUBRIC)

        output_filename = f"{uuid.uuid4()}_{os.path.splitext(file.filename)[0]}_rubric_output.txt"
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(result_text)

        os.remove(temp_pdf_path)

        return JSONResponse(content={
            "filename": output_filename,
            "content": result_text
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ------------------------------
# Generate Score Endpoint
# ------------------------------
@app.post("/generate_score")
async def generate_score(rubric_file: UploadFile = File(...), answer_file: UploadFile = File(...)):
    try:
        rubric_path = f"temp_{rubric_file.filename}"
        answer_path = f"temp_{answer_file.filename}"

        with open(rubric_path, "wb") as f:
            f.write(await rubric_file.read())
        with open(answer_path, "wb") as f:
            f.write(await answer_file.read())

        setup_auth()

        PROMPT_ANSWERSCRIPT = (
            "You are an expert transcriptionist specializing in handwritten documents."
            "Transcribe the attached PDF, which contains handwritten questions and answers."
            "Your task is to produce a clean, plain-text version of the content."
            "Follow these rules precisely:"
            "1. Preserve the question and answer (Q&A) format."
            "2. Start each question with the prefix 'Question:' on a new line."
            "3. Start each answer with the prefix 'Answer:' on a new line."
            "4. For any handwritten math, transcribe it into clear, readable LaTeX format (e.g., $E = mc^2$, $\\frac{a}{b}$)."
        )

        PROMPT_RUBRIC = (
            "You are an AI assistant specializing in educational assessment."
            "Analyze the attached PDF, which appears to be a scoring rubric or grading guide."
            "Your task is to extract and transcribe this rubric into a clean, plain-text format."
            "Preserve all scoring criteria, sub-criteria, and their associated point values."
            "Structure the output logically, clearly linking criteria to their points."
        )

        rubric_text = transcribe_pdf_from_path(rubric_path, PROMPT_RUBRIC)
        student_answer = transcribe_pdf_from_path(answer_path, PROMPT_ANSWERSCRIPT)

        result_text = grade_student_answer(rubric_text, student_answer)

        output_filename = f"{uuid.uuid4()}_score_output.txt"
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(result_text)

        os.remove(rubric_path)
        os.remove(answer_path)

        return JSONResponse(content={
            "filename": output_filename,
            "content": result_text
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ------------------------------
# Optional Root Endpoint
# ------------------------------
@app.get("/")
def read_root():
    return {"message": "FastAPI AI Graded Assignments Server is running!"}

# ------------------------------
# Grade all submissions for an assignment
# ------------------------------
@app.post("/grade/submissions")
async def grade_submissions(payload: Dict[str, Any] = Body(...)):
    try:
        assignment_id = payload.get("assignment_id")
        assignment_idea = payload.get("assignment_idea")

        if not assignment_id or not assignment_idea:
            raise HTTPException(status_code=400, detail="assignment_id and assignment_idea are required in the request body")

        graded = grade_submissions_for_assignment(assignment_id, assignment_idea)
        return JSONResponse(content=graded)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ------------------------------
# Final Grading Wrapper Endpoint
# ------------------------------
@app.post("/final_grading")
async def final_grading(payload: Dict[str, Any] = Body(...)):
    """
    Wrapper API to grade an assignment using 'grade_submissions_for_assignment'.
    Expects JSON body with:
      - assignment_id: the assignment identifier
    """
    try:
        assignment_id = payload.get("assignment_id")

        if not assignment_id:
            raise HTTPException(
                status_code=400,
                detail="assignment_id is required in the request body"
            )

        # Call existing helper function
        graded_results = grade_submissions_for_assignment(
            assignment_id=assignment_id
        )

        return JSONResponse(content={
            "message": "Final grading completed successfully",
            "graded_count": graded_results.get("count", 0),
            "results": graded_results.get("results", [])
        })

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


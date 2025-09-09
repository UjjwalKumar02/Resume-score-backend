import os
import io
import joblib
import numpy as np
import pandas as pd
from typing import Optional, List
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, UploadFile, File, Form
from app.similarity import calculate_tfidf_similarity, calculate_bert_similarity
from app.extraction import extract_text_from_pdf, extract_text_from_docx, extract_skills


app = FastAPI(title="Resume score backend")

origins = [
    "http://localhost:3000",
    "https://resume-score-2q5k.vercel.app"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Model unfolding
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
regressor_model_path = os.path.join(BASE_DIR, "../models/regressor_model.pkl")
classifier_model_path = os.path.join(BASE_DIR, "../models/classifier_model.pkl")
encoder_path = os.path.join(BASE_DIR, "../models/encoder.pkl")

regressor_model = joblib.load(regressor_model_path)
classifier_model = joblib.load(classifier_model_path)
label_encoder = joblib.load(encoder_path)


# API endpoint
@app.post("/score-prediction")
async def score_prediction(
    resume: UploadFile = File(...),
    jd_file: Optional[UploadFile] = File(None),
    jd_text_input: Optional[str] = Form(None),
):
    # Resume handling
    resume_bytes = await resume.read()
    if resume.filename.lower().endswith(".pdf"):
        resume_text = extract_text_from_pdf(resume_bytes)
    else:
        resume_text = extract_text_from_docx(io.BytesIO(resume_bytes))

    # JD handling
    if jd_file:
        jd_bytes = await jd_file.read()
        jd_text = (
            extract_text_from_pdf(jd_bytes)
            if jd_file.filename.lower().endswith(".pdf")
            else extract_text_from_docx(io.BytesIO(jd_bytes))
        )
    elif jd_text_input:
        jd_text = jd_text_input
    else:
        return {"error": "No JD provided!"}

    # Similarity calculations
    tfidf_score = calculate_tfidf_similarity(resume_text, jd_text)
    bert_score = calculate_bert_similarity(resume_text, jd_text)

    # Exract skills
    resume_skills_dict = extract_skills(resume_text)
    jd_skills_dict = extract_skills(jd_text)

    resume_skills = set(resume_skills_dict.keys())
    jd_skills = set(jd_skills_dict.keys())

    matched_skills = sorted(jd_skills & resume_skills)
    missing_skills = sorted(jd_skills - resume_skills)

    # Score prediction
    input_data_for_score = pd.DataFrame(
        [
            {
                "Tfidf_Similarity": float(np.round(tfidf_score, 2)),
                "Bert_Similarity": float(np.round(bert_score, 2)),
                "No_of_Matched_Skills": len(matched_skills),
                "No_of_Missing_Skills": len(missing_skills),
            }
        ]
    )
    score = regressor_model.predict(input_data_for_score)

    # Category Prediction
    input_data_for_category = pd.DataFrame(
        [
            {
                "Tfidf_Similarity": float(np.round(tfidf_score, 2)),
                "Bert_Similarity": float(np.round(bert_score, 2)),
                "No_of_Matched_Skills": len(matched_skills),
                "No_of_Missing_Skills": len(missing_skills),
                "Score": float(np.round(score[0], 2)),
            }
        ]
    )
    category = classifier_model.predict(input_data_for_category)
    category = label_encoder.inverse_transform(category)

    # Log respsonse
    log_response = {
        "resume_skills": list(resume_skills_dict.values()),
        "jd_skills": list(jd_skills_dict.values()),
        "Tfidf_Similarity": float(np.round(tfidf_score, 2)),
        "Bert_Similarity": float(np.round(bert_score, 2)),
    }

    return {
        "message": "Processed successfully",
        "data": {
            **log_response,
            "matched_skills": [resume_skills_dict[skill] for skill in matched_skills],
            "missing_skills": [jd_skills_dict[skill] for skill in missing_skills],
            "score": float(np.round(score[0], 0)),
            "category": category[0],
        },
    }


# API endpoint
@app.post("/rank-resumes")
async def rank_resumes(
    resumes: List[UploadFile] = File(...),
    jd_file: Optional[UploadFile] = File(None),
    jd_text_input: Optional[str] = Form(None),
):
    # JD handling
    if jd_file:
        jd_bytes = await jd_file.read()
        jd_text = (
            extract_text_from_pdf(jd_bytes)
            if jd_file.filename.lower().endswith(".pdf")
            else extract_text_from_docx(io.BytesIO(jd_bytes))
        )
    elif jd_text_input:
        jd_text = jd_text_input
    else:
        return {"error": "No JD provided!"}

    results = []

    for resume in resumes:
        resume_bytes = await resume.read()
        if resume.filename.lower().endswith(".pdf"):
            resume_text = extract_text_from_pdf(resume_bytes)
        else:
            resume_text = extract_text_from_docx(io.BytesIO(resume_bytes))

        # Similarity
        tfidf_score = calculate_tfidf_similarity(resume_text, jd_text)
        bert_score = calculate_bert_similarity(resume_text, jd_text)

        # Skills
        resume_skills_dict = extract_skills(resume_text)
        jd_skills_dict = extract_skills(jd_text)

        resume_skills = set(resume_skills_dict.keys())
        jd_skills = set(jd_skills_dict.keys())

        matched_skills = sorted(jd_skills & resume_skills)
        missing_skills = sorted(jd_skills - resume_skills)

        # Score prediction
        input_data_for_score = pd.DataFrame(
            [
                {
                    "Tfidf_Similarity": float(np.round(tfidf_score, 2)),
                    "Bert_Similarity": float(np.round(bert_score, 2)),
                    "No_of_Matched_Skills": len(matched_skills),
                    "No_of_Missing_Skills": len(missing_skills),
                }
            ]
        )
        score = regressor_model.predict(input_data_for_score)

        # Category prediction
        input_data_for_category = pd.DataFrame(
            [
                {
                    "Tfidf_Similarity": float(np.round(tfidf_score, 2)),
                    "Bert_Similarity": float(np.round(bert_score, 2)),
                    "No_of_Matched_Skills": len(matched_skills),
                    "No_of_Missing_Skills": len(missing_skills),
                    "Score": float(np.round(score[0], 2)),
                }
            ]
        )
        category = classifier_model.predict(input_data_for_category)
        category = label_encoder.inverse_transform(category)

        results.append(
            {
                "resume_name": resume.filename,
                "score": float(np.round(score[0], 0)),
                "category": category[0],
                "matched_skills": [
                    resume_skills_dict[skill] for skill in matched_skills
                ],
                "missing_skills": [jd_skills_dict[skill] for skill in missing_skills],
                "Tfidf_Similarity": float(np.round(tfidf_score, 2)),
                "Bert_Similarity": float(np.round(bert_score, 2))
            }
        )

    results = sorted(results, key=lambda x: x["score"], reverse=True)

    return {"message": "Processed successfully", "results": results}

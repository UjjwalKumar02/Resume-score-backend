import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def calculate_tfidf_similarity(resume_text: str, jd_text: str) -> float:
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform([resume_text, jd_text])
    return cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]


def calculate_jaccard_similarity(resume_text: str, jd_text: str) -> float:
    resume_tokens = set(resume_text.lower().split())
    jd_tokens = set(jd_text.lower().split())

    intersection = resume_tokens & jd_tokens
    union = resume_tokens | jd_tokens

    if not union:
        return 0.0
    return len(intersection) / len(union)


def calculate_length_ratio(resume_text: str, jd_text: str) -> float:
    resume_len = len(resume_text)
    jd_len = len(jd_text)
    if jd_len == 0:
        return 0.0
    return round(resume_len / jd_len, 2)

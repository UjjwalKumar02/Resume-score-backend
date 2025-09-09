import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load BERT model for embeddings
bert_model = SentenceTransformer("all-MiniLM-L6-v2")


def calculate_tfidf_similarity(resume_text: str, jd_text: str) -> float:
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform([resume_text, jd_text])
    return cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]


def calculate_bert_similarity(resume_text, jd_text):
    """Returns cosine similarity between two texts using BERT embeddings."""
    embeddings = bert_model.encode([resume_text, jd_text], convert_to_tensor=True)
    return cosine_similarity(
        embeddings[0].unsqueeze(0).cpu().numpy(),
        embeddings[1].unsqueeze(0).cpu().numpy(),
    )[0][0]

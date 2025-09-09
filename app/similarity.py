import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from sentence_transformers import SentenceTransformer

# Use a smaller model to save memory
MODEL_NAME = "paraphrase-MiniLM-L3-v2"  # much smaller than L6-v2

_bert_model = None  # Global variable to cache the model


def get_bert_model():
    global _bert_model
    if _bert_model is None:
        _bert_model = SentenceTransformer(MODEL_NAME)
    return _bert_model


def calculate_tfidf_similarity(resume_text: str, jd_text: str) -> float:
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform([resume_text, jd_text])
    return cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]


def calculate_bert_similarity(resume_text, jd_text):
    """Returns cosine similarity between two texts using BERT embeddings."""
    model = get_bert_model()
    embeddings = model.encode([resume_text, jd_text], convert_to_tensor=True)

    # Avoid tensor operations to save memory
    emb1 = embeddings[0].cpu().numpy().reshape(1, -1)
    emb2 = embeddings[1].cpu().numpy().reshape(1, -1)
    
    return cosine_similarity(emb1, emb2)[0][0]

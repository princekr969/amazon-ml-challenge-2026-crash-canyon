"""Embedding-based blocking using sentence-transformers."""
import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer


def encode_names(names, model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
    """Encode a list of names using a sentence-transformer model."""
    model = SentenceTransformer(model_name)
    return model.encode(names.tolist(), normalize_embeddings=True, 
                        batch_size=256, show_progress_bar=True)


def build_faiss_index(embeddings: np.ndarray):
    """Build a FAISS index for fast similarity search."""
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings.astype('float32'))
    return index


def embedding_block(s1_emb, s2_emb, s1_ids, s2_ids, top_k=100):
    """Return top-k candidates per S1 by embedding cosine similarity."""
    index = build_faiss_index(s2_emb)
    scores, idx = index.search(s1_emb.astype('float32'), top_k)
    pairs = set()
    for i, s1_id in enumerate(s1_ids):
        for j, score in zip(idx[i], scores[i]):
            if j >= 0 and score > 0.5:
                pairs.add((s1_id, s2_ids[j]))
    return pairs
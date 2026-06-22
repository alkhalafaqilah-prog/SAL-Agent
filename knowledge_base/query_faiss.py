import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

# Load model, index and documents
model = SentenceTransformer("all-mpnet-base-v2")
index = faiss.read_index("knowledge_base/beamdata.index")

with open("knowledge_base/documents.pkl", "rb") as f:
    documents = pickle.load(f)

def query_kb(query: str, top_k: int = 1):
    """
    Takes a query string, returns the top_k most
    relevant KB entries from the FAISS index.
    """
    # Convert query to embedding
    query_embedding = model.encode([query])
    query_embedding = np.array(query_embedding, dtype="float32")

    # Search FAISS
    distances, indices = index.search(query_embedding, k=top_k)

    results = []
    for i, idx in enumerate(indices[0]):
        results.append({
            "rank":     i + 1,
            "kb_id":    documents[idx]["kb_id"],
            "name":     documents[idx]["name"],
            "category": documents[idx]["category"],
            "score":    round(float(distances[0][i]), 4),
            "text":     documents[idx]["text"],
        })

    return results


# Test it
if __name__ == "__main__":
    test_queries = [
        "Client not ready for AI",
        "Manufacturing company wants automation",
        "Healthcare company needs secure internal knowledge retrieval",
        "Client thinks ChatGPT is enough",
        "CTO interested in enterprise AI governance",
    ]

    for query in test_queries:
        print(f"\nQuery: {query}")
        results = query_kb(query, top_k=1)
        for r in results:
            print(f"  → {r['kb_id']} | {r['name']} | score: {r['score']}")
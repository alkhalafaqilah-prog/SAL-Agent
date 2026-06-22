import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import os

# Load model
model = SentenceTransformer("all-mpnet-base-v2")

# Load FAISS index & documents
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

index = faiss.read_index(
    os.path.join(BASE_DIR, "project.index")
)

with open(os.path.join(BASE_DIR, "project_documents.pkl"), "rb") as f:
    documents = pickle.load(f)

print(f"Projects index loaded — {index.ntotal} entries")

# -----------------------------
# Retrieval function
# -----------------------------
def retrieve_project_context(query, top_k=1):
    """
    Takes a query string, returns top_k most
    relevant past projects from the FAISS index.
    Score is cosine similarity — higher = better match.
    """
    query_embedding = model.encode(
        [query],
        normalize_embeddings=True
    )
    query_embedding = np.array(
        query_embedding,
        dtype="float32"
    )

    scores, indices = index.search(query_embedding, top_k)

    results = []
    for i, idx in enumerate(indices[0]):
        doc = documents[idx].copy()
        doc["score"] = round(float(scores[0][i]), 4)
        results.append(doc)

    return results

# Test queries — one per industry
TEST_QUERIES = [
    {
        "label":    "Healthcare",
        "query":    "Healthcare AI patient assistant secure medical solution",
    },
    {
        "label":    "EdTech",
        "query":    "EdTech learning platform student AI personalized education",
    },
    {
        "label":    "Manufacturing",
        "query":    "Manufacturing predictive maintenance fleet performance monitoring",
    },
    {
        "label":    "Finance",
        "query":    "Finance loan approval automation credit risk assessment",
    },
    {
        "label":    "Transportation",
        "query":    "Transportation driver marketplace logistics fleet distribution",
    },
]

print("\n" + "="*60)
print("PROJECT RAG — Query Test")
print("="*60)

all_passed = True

for t in TEST_QUERIES:
    print(f"\nIndustry: {t['label']}")
    print(f"Query: {t['query']}")
    print()

    results = retrieve_project_context(t["query"], top_k=1)

    for r in results:
        print(f"  → {r['project_id']} | {r['project_name']} | score: {r['score']}")

    # Warn if top score is too low (below 0.30 = weak match)
    if results and results[0]["score"] < 0.30:
        print(f"Low confidence match for {t['label']}")
        all_passed = False

    print("-"*60)

print()
if all_passed:
    print("All queries returned confident matches — RAG ready")
else:
    print("Some queries returned weak matches — review flagged industries")
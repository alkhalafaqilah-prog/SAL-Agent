import pandas as pd
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

# Load embedding model
model = SentenceTransformer("all-mpnet-base-v2")

# Load BeamData knowledge base CSV
df = pd.read_csv("knowledge_base/beamdata_knowledge_base_cleaned.csv")
df = df.fillna("")  # prevent nan in text fields

documents = []

# Convert rows into searchable text
for _, row in df.iterrows():

    text = f"""
    Category: {row['category']}
    Name: {row['name']}
    Type: {row['type']}
    Tagline: {row['tagline']}
    Description: {row['description']}
    Key Features: {row['key_features']}
    Target Industries: {row['target_industries']}
    Target Roles: {row['target_roles']}
    Pain Points Solved: {row['pain_points_solved']}
    Ideal Customer Profile: {row['ideal_customer_profile']}
    Differentiators: {row['differentiators']}
    Deployment Options: {row['deployment_options']}
    Compliance: {row['compliance']}
    Pricing Model: {row['pricing_model']}
    Proof Points: {row['proof_points']}
    Objection 1: {row['objection_1']} — {row['objection_1_response']}
    Objection 2: {row['objection_2']} — {row['objection_2_response']}
    Objection 3: {row['objection_3']} — {row['objection_3_response']}
    Competitor Context: {row['competitor_context']}
    CTA: {row['cta']}
    """

    # ONE append only — dict with text inside
    documents.append({
        "kb_id":    row['kb_id'],
        "name":     row['name'],
        "category": row['category'],
        "text":     text
    })

# Extract plain text strings for encoding
texts = [d["text"] for d in documents]

# Create embeddings from text strings only
embeddings = model.encode(texts, show_progress_bar=True)
embeddings = np.array(embeddings, dtype="float32")

# Create FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# Save FAISS index
faiss.write_index(index, "knowledge_base/beamdata.index")

# Save full document dicts for retrieval
with open("knowledge_base/documents.pkl", "wb") as f:
    pickle.dump(documents, f)

print(f"FAISS index built — {len(documents)} entries, dimension {dimension}")



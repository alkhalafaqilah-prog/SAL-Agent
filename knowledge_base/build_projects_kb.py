from markitdown import MarkItDown
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import numpy as np
import os

# Load model
model = SentenceTransformer("all-mpnet-base-v2")

# Convert PDF → text
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PDF_PATH = os.path.join(BASE_DIR, "Beamdata_Past_Project_Descriptions.pdf")

md = MarkItDown()
result = md.convert(PDF_PATH)
markdown_text = result.text_content

# Split by detecting title lines
SKIP_STARTS = [
    'problem', 'solution', 'result', 'heading', 'summary',
    'testimonial', 'the ', 'this ', 'by ', 'with ', 'for ',
    'using ', 'our ', 'we ', 'a ', 'an ', 'in ', 'key ',
    'tech', 'tool', '●', '○', '-', '*', 'http',
    'major', 'research', 'conclusion', 'infra', 'analytics',
    'applied', 'information', 'top ', 'identified', 'improved',
    'prepared', 'fangraphs', 'deepar', 'statistics', 'agentic',
    'athena', 'sku ', 'users)', 'singapore', 'hockey',
    'jupyter', 'teck', 'build ', 'these ', 'future ',
    'question', 'canada,', 'identifying', 'python (', 'scikit',
]

lines = markdown_text.split('\n')
documents = []
current_title = None
current_lines = []

for line in lines:
    s = line.strip()
    if not s:
        continue
    
    if current_title and current_title.endswith('&'):
        current_title = current_title + ' ' + s
        continue

    is_title = (
        0 < len(s) < 80
        and s[0].isupper()
        and not s.endswith(':')
        and not s.endswith('.')
        and not s.endswith(',')
        and not s.endswith('"')
        and not s.endswith('with')
        and len(s.split()) >= 2
        and not any(s.lower().startswith(k) for k in SKIP_STARTS)
    )

    if is_title and current_title and len(current_lines) > 5:
        content = '\n'.join(current_lines).strip()
        if len(content) > 100:
            documents.append({
                "project_id":   f"BD-PROJ-{len(documents)+1:03d}",
                "project_name": current_title,
                "source":       "beamdata_projects_pdf",
                "text":         f"{current_title}\n\n{content}"
            })
        current_title = s
        current_lines = []
    else:
        current_lines.append(s)
        if current_title is None and 0 < len(s) < 80:
            current_title = s

# Save last project
if current_title and len(current_lines) > 5:
    content = '\n'.join(current_lines).strip()
    if len(content) > 100:
        documents.append({
            "project_id":   f"BD-PROJ-{len(documents)+1:03d}",
            "project_name": current_title,
            "source":       "beamdata_projects_pdf",
            "text":         f"{current_title}\n\n{content}"
        })

print(f"Extracted {len(documents)} project chunks:")
for d in documents:
    print(f"  {d['project_id']} | {d['project_name']}")


# Create embeddings
texts = [d["text"] for d in documents]

embeddings = model.encode(
    texts,
    normalize_embeddings=True,
    show_progress_bar=True
)
embeddings = np.array(embeddings, dtype="float32")

# Create FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)
index.add(embeddings)

# Save
INDEX_PATH = os.path.join(BASE_DIR, "project.index")
DOCS_PATH  = os.path.join(BASE_DIR, "project_documents.pkl")

faiss.write_index(index, INDEX_PATH)

with open(DOCS_PATH, "wb") as f:
    pickle.dump(documents, f)

print(f"\nProject RAG built — {len(documents)} chunks | dimension {dimension}")


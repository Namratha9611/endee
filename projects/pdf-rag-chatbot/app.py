import os
import json
from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
import numpy as np
from pypdf import PdfReader
from endee import Endee, Precision

app = Flask(__name__)

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")
DIMENSION = 384
INDEX_NAME = "pdf_rag_index"
LOCAL_STORAGE_FILE = "local_vec_store.json"

# Initialize Endee Client
# Ensure the Endee server is running (default port 8080)
index = None
LOCAL_MODE = False
local_data = []

try:
    client = Endee()
    # Create index if it doesn't exist
    try:
        client.create_index(
            name=INDEX_NAME,
            dimension=DIMENSION,
            space_type="cosine",
            precision=Precision.INT8D
        )
        print(f"Connected to Endee. Index '{INDEX_NAME}' created/verified.")
    except Exception as e:
        print(f"Endee connection note: {e}")
    
    index = client.get_index(name=INDEX_NAME)
    print("Endee Vector DB is ACTIVE.")
except Exception as e:
    print(f"WARNING: Failed to connect to Endee: {e}")
    print("Switching to LOCAL FALLBACK MODE (saving to JSON).")
    LOCAL_MODE = True
    if os.path.exists(LOCAL_STORAGE_FILE):
        with open(LOCAL_STORAGE_FILE, 'r') as f:
            local_data = json.load(f)

@app.route("/")
def home():
    status = "Endee DB" if not LOCAL_MODE else "Local Fallback (JSON)"
    return f"PDF RAG Chatbot - Server Running (Mode: {status})"

@app.route("/upload-pdf", methods=["POST"])
def upload_pdf():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    if not file.filename.endswith(".pdf"):
        return jsonify({"error": "Only PDF files are supported"}), 400

    # Read PDF
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"

    # Simple chunking (by paragraph)
    chunks = [c.strip() for c in text.split("\n\n") if c.strip()]
    
    if not chunks:
        return jsonify({"error": "No text extracted from PDF"}), 400

    # Generate embeddings
    vector_items = []
    for i, chunk in enumerate(chunks):
        embedding = model.encode(chunk).tolist()
        item = {
            "id": f"{file.filename}_{i}",
            "vector": embedding,
            "meta": {"text": chunk, "filename": file.filename}
        }
        vector_items.append(item)

    if not LOCAL_MODE and index:
        index.upsert(vector_items)
        return jsonify({"message": f"Successfully stored {len(chunks)} chunks in Endee."})
    else:
        # Local Fallback path
        local_data.extend(vector_items)
        with open(LOCAL_STORAGE_FILE, 'w') as f:
            json.dump(local_data, f)
        return jsonify({"message": f"Successfully stored {len(chunks)} chunks in LOCAL STORAGE."})

@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    question = data["question"]

    # Generate query embedding
    q_embedding = model.encode(question).tolist()
    
    if not LOCAL_MODE and index:
        # Query Endee
        results = index.query(vector=q_embedding, top_k=3)
        if not results:
            return jsonify({"answer": "No relevant information found."})
        
        answers = []
        for res in results:
            meta = getattr(res, 'meta', {}) if not isinstance(res, dict) else res.get('meta', {})
            answers.append(meta.get('text', ''))
    else:
        # Local Semantic Search (Cosine Similarity)
        if not local_data:
            return jsonify({"answer": "No documents uploaded yet."})
        
        scores = []
        for item in local_data:
            vec = np.array(item['vector'])
            q_vec = np.array(q_embedding)
            score = np.dot(vec, q_vec) / (np.linalg.norm(vec) * np.linalg.norm(q_vec))
            scores.append((score, item['meta']['text']))
        
        # Sort by score and take top 3
        scores.sort(key=lambda x: x[0], reverse=True)
        answers = [s[1] for s in scores[:3]]

    return jsonify({
        "answer": "\n\n".join(answers)
    })

if __name__ == "__main__":
    app.run(debug=True, port=5000)

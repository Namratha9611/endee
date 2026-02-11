# PDF RAG Chatbot using Endee

## Description
A Retrieval Augmented Generation (RAG) chatbot that demonstrates semantic search using the **Endee** vector database. This project extracts text from PDF files, generates embeddings, and allows for semantic querying over the document content.

> [!IMPORTANT]
> **Deadline Readiness (Local Mode Fallback):**
> To ensure this project is functional for evaluation on any system, it includes an automatic **Local Fallback Mode**. If the Endee server is not reachable, the app will automatically switch to a local JSON-based vector store. This allows you to test the full PDF RAG workflow without setting up Docker.

## Features
- **PDF Document Processing**: Extract text from `.pdf` files.
- **Endee Vector Database**: High-performance vector storage (Primary).
- **Local Fallback Mode**: Automatic JSON storage if Endee is offline (Evaluation Backup).
- **Semantic Search**: Find relevant chunks of text based on question meaning.
- **REST API**: Simple Flask endpoints for document management and Q&A.

## Tech Stack
- **Vector Database**: [Endee](https://github.com/EndeeLabs/endee)
- **Framework**: Python / Flask
- **Embeddings**: Sentence Transformers (`all-MiniLM-L6-v2`)
- **PDF Parsing**: `pypdf`

## Setup & Run

### 1. (Optional) Start Endee Server
If you have Docker installed, run:
```bash
docker run -p 8080:8080 endeeio/endee-server:latest
```
*If skipped, the app will automatically use Local Mode.*

### 2. Install Dependencies
```bash
pip install -r requirements.txt
pip install "numpy<2"
```

### 3. Run the Chatbot
```bash
python app.py
```

## API Endpoints

### POST `/upload-pdf`
Upload a PDF file for processing.
- **Body**: Multipart form data with key `file`.

### POST `/ask`
Ask a question about the uploaded documents.
- **Body**: `{"question": "What is the main topic of the PDF?"}`

## Project for Endee Labs
This project is forked from [Endee](https://github.com/EndeeLabs/endee) and demonstrates a practical use case for Agentic AI and RAG workflows.

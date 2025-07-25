import os
import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from ragbi import RAGSystem
import yaml

# Initialize RAG system
rag = RAGSystem()
app = FastAPI(title="RAG API", version="1.0")

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load API keys
with open('gemini_api.yml', 'r') as file:
    api_creds = yaml.safe_load(file)
    os.environ['GOOGLE_API_KEY'] = api_creds['api']['api_key']

# --- API Models ---
class QueryRequest(BaseModel):
    text: str

class SourceItem(BaseModel):
    source_file: str
    pages: str
    content_excerpt: str

class ResponseModel(BaseModel):
    answer: str
    sources: list[SourceItem]

# --- API Endpoints ---
@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    file_bytes = await file.read()
    chunks_added = rag.process_upload(file_bytes, file.filename)
    return {
        "filename": file.filename,
        "chunks_added": chunks_added,
        "status": "success" if chunks_added > 0 else "failed"
    }

@app.post("/query", response_model=ResponseModel)
async def process_query(request: QueryRequest):
    answer, docs = rag.get_response(request.text)
    sources = []
    for doc in docs:
        sources.append(SourceItem(
            source_file=doc.metadata.get("source", "Unknown"),
            pages=doc.metadata.get("pages", "N/A"),
            content_excerpt=doc.page_content[:150] + "..." if len(doc.page_content) > 150 else doc.page_content
        ))
    return ResponseModel(answer=answer, sources=sources)

@app.get("/health")
def health_check():
    return {
        "status": "active",
        "processed_files": len(rag.processed_files),
        "total_chunks": len(rag.all_docs)
    }

if __name__ == "__main__":
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        log_level="info"
    )

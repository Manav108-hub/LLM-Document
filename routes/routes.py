from fastapi import APIRouter, File, UploadFile, HTTPException, Request
from fastapi.responses import JSONResponse, HTMLResponse
from pathlib import Path
import aiofiles
import os
from typing import List

from core.models import QueryRequest, QueryResponse, DocumentInfo
from config import config

router = APIRouter()

@router.get("/", response_class=HTMLResponse)
async def read_root():
    """Serve the main web interface"""
    with open("templates/index.html", "r") as f:
        return HTMLResponse(content=f.read())

@router.post("/upload-document/")
async def upload_document(request: Request, file: UploadFile = File(...)):
    """Upload and process a document"""
    ai_agent = request.app.state.ai_agent
    
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    
    # Check file extension
    file_extension = Path(file.filename).suffix.lower()
    
    if file_extension not in config.ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported file type. Allowed: {', '.join(config.ALLOWED_EXTENSIONS)}"
        )
    
    # Save uploaded file
    file_path = config.UPLOAD_DIR / file.filename
    
    async with aiofiles.open(file_path, 'wb') as f:
        content = await file.read()
        
        # Check file size
        if len(content) > config.MAX_FILE_SIZE:
            raise HTTPException(status_code=400, detail="File too large")
            
        await f.write(content)
    
    # Process the document
    try:
        result = await ai_agent.process_document(str(file_path), file.filename)
        return result
    except Exception as e:
        # Clean up file if processing failed
        if file_path.exists():
            file_path.unlink()
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/query/", response_model=QueryResponse)
async def query_documents(request: Request, query_request: QueryRequest):
    """Query the uploaded documents"""
    ai_agent = request.app.state.ai_agent
    
    if not query_request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    result = await ai_agent.generate_response(query_request.query)
    return QueryResponse(**result)

@router.get("/documents/", response_model=List[DocumentInfo])
async def get_documents(request: Request):
    """Get list of uploaded documents"""
    ai_agent = request.app.state.ai_agent
    return ai_agent.get_document_info()

@router.post("/clear-data/")
async def clear_all_data(request: Request):
    """Clear all processed data"""
    ai_agent = request.app.state.ai_agent
    ai_agent.clear_all_data()
    return {"status": "success", "message": "All data cleared"}

@router.get("/health/")
async def health_check(request: Request):
    """Health check endpoint"""
    ai_agent = request.app.state.ai_agent
    return {
        "status": "healthy",
        "documents_count": len(ai_agent.documents),
        "chunks_count": len(ai_agent.document_chunks),
        "device": ai_agent.device
    }
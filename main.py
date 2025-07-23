from fastapi import FastAPI
from routes.routes import router
from core.ai_agent import DocumentAIAgent
import uvicorn

# Initialize the AI agent (singleton)
ai_agent = DocumentAIAgent()

# FastAPI app
app = FastAPI(
    title="Document AI Agent",
    description="Upload documents and ask questions based on their content",
    version="1.0.0"
)

# Include routes
app.include_router(router)

# Make ai_agent available to routes
app.state.ai_agent = ai_agent

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
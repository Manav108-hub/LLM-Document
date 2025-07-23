# Document AI Agent

A high-accuracy document-based AI agent built with FastAPI that allows you to upload documents and ask questions based on their content.

## Features

- **High-Accuracy Models**: Uses BAAI/bge-large-en-v1.5 for embeddings and DialoGPT-large for generation
- **Multiple Document Formats**: Supports PDF, DOCX, and TXT files
- **Smart Chunking**: Intelligent text splitting with overlap for better context preservation
- **Vector Search**: FAISS-based similarity search for relevant content retrieval
- **Web Interface**: Clean, intuitive web UI
- **Confidence Scoring**: Provides reliability metrics for responses
- **Source Attribution**: Shows which documents contributed to answers
- **Persistent Storage**: Saves processed data for quick restart

## Models Used

### Embedding Model: BAAI/bge-large-en-v1.5
- **Size**: ~1.34GB
- **Embedding Dimension**: 1024
- **Strengths**: State-of-the-art semantic understanding, excellent for retrieval tasks
- **Performance**: High accuracy with comprehensive context capture

### Generation Model: Microsoft/DialoGPT-large
- **Size**: ~1.5GB  
- **Context Length**: 2048 tokens
- **Strengths**: Natural conversational responses, good coherence
- **Performance**: Balanced between quality and speed

## Installation & Setup

1. **Clone/Create the project structure**:
```bash
mkdir document_ai_agent
cd document_ai_agent
import os
from pathlib import Path

class Config:
    # Model configuration
    EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"
    GENERATION_MODEL = "microsoft/DialoGPT-large"
    
    # Hyperparameters
    CHUNK_SIZE = 800
    CHUNK_OVERLAP = 100
    TOP_K_RETRIEVAL = 7
    MAX_RESPONSE_LENGTH = 1024
    TEMPERATURE = 0.3
    
    # Directories
    UPLOAD_DIR = Path("uploads")
    DATA_DIR = Path("data")
    
    # Device
    DEVICE = "cuda" if os.system("nvidia-smi") == 0 else "cpu"
    
    # File settings
    ALLOWED_EXTENSIONS = {'.pdf', '.docx', '.txt'}
    MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
    
    def __init__(self):
        # Create directories
        self.UPLOAD_DIR.mkdir(exist_ok=True)
        self.DATA_DIR.mkdir(exist_ok=True)

config = Config()
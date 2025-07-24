import os
import torch
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
    
    # Device detection with better compatibility
    @property
    def DEVICE(self):
        """Dynamically determine the best device to use"""
        if torch.cuda.is_available():
            try:
                # Test if CUDA actually works
                torch.cuda.current_device()
                return "cuda"
            except Exception:
                return "cpu"
        return "cpu"
    
    # File settings
    ALLOWED_EXTENSIONS = {'.pdf', '.docx', '.txt'}
    MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
    
    def __init__(self):
        # Create directories
        self.UPLOAD_DIR.mkdir(exist_ok=True)
        self.DATA_DIR.mkdir(exist_ok=True)
        
        # Log device info
        device = self.DEVICE
        print(f"Using device: {device}")
        if device == "cuda":
            print(f"CUDA device: {torch.cuda.get_device_name()}")
            print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

config = Config()
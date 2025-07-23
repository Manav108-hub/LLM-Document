import logging
import time
from functools import wraps
from pathlib import Path
from typing import Dict, Any
import hashlib

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('app.log'),
            logging.StreamHandler()
        ]
    )

def timer(func):
    """Decorator to measure function execution time"""
    @wraps(func)
    async def async_wrapper(*args, **kwargs):
        start_time = time.time()
        result = await func(*args, **kwargs)
        end_time = time.time()
        logging.info(f"{func.__name__} took {end_time - start_time:.2f} seconds")
        return result
    
    @wraps(func)
    def sync_wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logging.info(f"{func.__name__} took {end_time - start_time:.2f} seconds")
        return result
    
    if hasattr(func, '__code__') and 'await' in func.__code__.co_names:
        return async_wrapper
    return sync_wrapper

def get_file_hash(file_path: str) -> str:
    """Generate hash for file to detect duplicates"""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def validate_file_size(file_size: int, max_size: int = 50 * 1024 * 1024) -> bool:
    """Validate file size (default 50MB)"""
    return file_size <= max_size

def clean_text(text: str) -> str:
    """Clean and normalize text"""
    import re
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove special characters but keep basic punctuation
    text = re.sub(r'[^\w\s.,!?;:()\-]', '', text)
    return text.strip()

def format_response_sources(sources: list) -> str:
    """Format sources for better display"""
    if not sources:
        return "No sources found"
    
    formatted = []
    for i, source in enumerate(sources, 1):
        formatted.append(f"{i}. {source}")
    
    return "\n".join(formatted)

class ResponseFormatter:
    """Format AI responses for better readability"""
    
    @staticmethod
    def format_confidence(confidence: float) -> str:
        """Format confidence score with descriptive text"""
        percentage = confidence * 100
        if percentage >= 90:
            return f"{percentage:.1f}% (Very High)"
        elif percentage >= 70:
            return f"{percentage:.1f}% (High)"
        elif percentage >= 50:
            return f"{percentage:.1f}% (Medium)"
        elif percentage >= 30:
            return f"{percentage:.1f}% (Low)"
        else:
            return f"{percentage:.1f}% (Very Low)"
    
    @staticmethod
    def format_processing_time(seconds: float) -> str:
        """Format processing time in human-readable format"""
        if seconds < 1:
            return f"{seconds*1000:.0f}ms"
        elif seconds < 60:
            return f"{seconds:.1f}s"
        else:
            minutes = int(seconds // 60)
            remaining_seconds = seconds % 60
            return f"{minutes}m {remaining_seconds:.1f}s"
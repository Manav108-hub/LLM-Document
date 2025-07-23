import re
from typing import List, Dict
from core.models import ChunkInfo
from config import config
import logging

logger = logging.getLogger(__name__)

class DocumentProcessor:
    @staticmethod
    def chunk_text(text: str, filename: str) -> List[ChunkInfo]:
        """Split text into overlapping chunks with metadata"""
        # Clean text
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Split into sentences for better chunking
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        chunks = []
        current_chunk = ""
        chunk_id = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            # Check if adding this sentence exceeds chunk size
            if len(current_chunk) + len(sentence) + 1 > config.CHUNK_SIZE:
                if current_chunk and len(current_chunk.strip()) > 100:
                    chunks.append(ChunkInfo(
                        text=current_chunk.strip(),
                        filename=filename,
                        chunk_id=chunk_id,
                        char_start=len(text) - len(current_chunk),
                        char_end=len(text)
                    ))
                    chunk_id += 1
                    
                    # Handle overlap
                    if config.CHUNK_OVERLAP > 0:
                        words = current_chunk.split()
                        if len(words) > config.CHUNK_OVERLAP:
                            overlap_words = words[-config.CHUNK_OVERLAP:]
                            current_chunk = " ".join(overlap_words) + " " + sentence
                        else:
                            current_chunk = sentence
                    else:
                        current_chunk = sentence
                else:
                    current_chunk = sentence
            else:
                current_chunk += " " + sentence if current_chunk else sentence
        
        # Add the last chunk
        if current_chunk.strip() and len(current_chunk.strip()) > 100:
            chunks.append(ChunkInfo(
                text=current_chunk.strip(),
                filename=filename,
                chunk_id=chunk_id,
                char_start=len(text) - len(current_chunk),
                char_end=len(text)
            ))
        
        return chunks
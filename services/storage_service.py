import json
import torch
import faiss
import os
from pathlib import Path
from typing import Dict, List, Any
import logging
from config import config

logger = logging.getLogger(__name__)

class StorageService:
    @staticmethod
    def save_data(documents: List[Dict], chunks: List[str], metadata: List[Dict]):
        """Save processed data to disk"""
        try:
            data = {
                'documents': documents,
                'document_chunks': chunks,
                'chunk_metadata': metadata
            }
            
            with open(config.DATA_DIR / 'processed_data.json', 'w') as f:
                json.dump(data, f, indent=2)
                
            logger.info("Data saved successfully")
        except Exception as e:
            logger.error(f"Error saving data: {e}")

    @staticmethod
    def load_data() -> tuple:
        """Load existing processed data"""
        try:
            data_file = config.DATA_DIR / 'processed_data.json'
            if data_file.exists():
                with open(data_file, 'r') as f:
                    data = json.load(f)
                
                documents = data.get('documents', [])
                chunks = data.get('document_chunks', [])
                metadata = data.get('chunk_metadata', [])
                
                logger.info(f"Loaded existing data: {len(documents)} documents, {len(chunks)} chunks")
                return documents, chunks, metadata
                
        except Exception as e:
            logger.error(f"Error loading existing data: {e}")
            
        return [], [], []

    @staticmethod
    def save_embeddings(embeddings: torch.Tensor):
        """Save embeddings to disk"""
        try:
            torch.save(embeddings, config.DATA_DIR / 'embeddings.pt')
        except Exception as e:
            logger.error(f"Error saving embeddings: {e}")

    @staticmethod
    def load_embeddings() -> torch.Tensor:
        """Load embeddings from disk"""
        try:
            embeddings_file = config.DATA_DIR / 'embeddings.pt'
            if embeddings_file.exists():
                return torch.load(embeddings_file, map_location='cpu')
        except Exception as e:
            logger.error(f"Error loading embeddings: {e}")
        return None

    @staticmethod
    def save_faiss_index(index: faiss.Index):
        """Save FAISS index to disk"""
        try:
            faiss.write_index(index, str(config.DATA_DIR / 'faiss_index.bin'))
        except Exception as e:
            logger.error(f"Error saving FAISS index: {e}")

    @staticmethod
    def load_faiss_index() -> faiss.Index:
        """Load FAISS index from disk"""
        try:
            index_file = config.DATA_DIR / 'faiss_index.bin'
            if index_file.exists():
                return faiss.read_index(str(index_file))
        except Exception as e:
            logger.error(f"Error loading FAISS index: {e}")
        return None

    @staticmethod
    def clear_all_data():
        """Clear all processed data"""
        files_to_remove = [
            'processed_data.json',
            'embeddings.pt',
            'faiss_index.bin'
        ]
        
        for filename in files_to_remove:
            file_path = config.DATA_DIR / filename
            if file_path.exists():
                file_path.unlink()
        
        # Clear uploads
        for file_path in config.UPLOAD_DIR.glob("*"):
            if file_path.is_file():
                file_path.unlink()
                
        logger.info("All data cleared")
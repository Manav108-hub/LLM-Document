import torch
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from typing import List
import logging
from config import config

logger = logging.getLogger(__name__)

class EmbeddingService:
    def __init__(self):
        self.model = None
        self.device = config.DEVICE

    def initialize_model(self):
        """Lazy loading of embedding model"""
        if self.model is None:
            logger.info("Loading embedding model...")
            self.model = SentenceTransformer(config.EMBEDDING_MODEL)
            self.model.to(self.device)

    def generate_embeddings(self, texts: List[str]) -> torch.Tensor:
        """Generate embeddings for a list of texts"""
        self.initialize_model()
        
        embeddings_list = []
        batch_size = 16
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            batch_embeddings = self.model.encode(
                batch,
                convert_to_tensor=True,
                show_progress_bar=False,
                batch_size=8
            )
            embeddings_list.append(batch_embeddings.cpu())
        
        return torch.cat(embeddings_list, dim=0)

    def create_faiss_index(self, embeddings: torch.Tensor) -> faiss.Index:
        """Create and populate FAISS index"""
        embeddings_np = embeddings.numpy().astype('float32')
        dimension = embeddings_np.shape[1]
        
        # Use IndexHNSWFlat for better accuracy
        index = faiss.IndexHNSWFlat(dimension, 32)
        index.hnsw.efConstruction = 200
        index.hnsw.efSearch = 100
        
        # Normalize vectors for cosine similarity
        faiss.normalize_L2(embeddings_np)
        index.add(embeddings_np)
        
        return index

    def search_similar(self, query: str, index: faiss.Index, k: int = 5) -> tuple:
        """Search for similar chunks"""
        self.initialize_model()
        
        query_embedding = self.model.encode([query], convert_to_tensor=True)
        query_np = query_embedding.cpu().numpy().astype('float32')
        faiss.normalize_L2(query_np)
        
        scores, indices = index.search(query_np, k)
        return scores[0], indices[0]
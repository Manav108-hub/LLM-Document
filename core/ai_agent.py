import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from typing import List, Dict
import re
import logging
from datetime import datetime

from config import config
from core.models import DocumentInfo, RetrievalResult
from core.document_processor import DocumentProcessor
from services.text_extraction import TextExtractionService
from services.embedding_service import EmbeddingService
from services.storage_service import StorageService

logger = logging.getLogger(__name__)

class DocumentAIAgent:
    def __init__(self):
        self.device = config.DEVICE
        
        # Services
        self.embedding_service = EmbeddingService()
        self.text_extractor = TextExtractionService()
        self.document_processor = DocumentProcessor()
        self.storage = StorageService()
        
        # Generation model components
        self.generator = None
        self.tokenizer = None
        
        # Document storage
        self.documents = []
        self.document_chunks = []
        self.chunk_metadata = []
        self.embeddings = None
        self.faiss_index = None
        
        # Load existing data
        self._load_existing_data()
        
        logger.info(f"DocumentAIAgent initialized with device: {self.device}")

    def initialize_generation_model(self):
        """Lazy loading of generation model"""
        if self.generator is None:
            logger.info("Loading generation model...")
            self.tokenizer = AutoTokenizer.from_pretrained(config.GENERATION_MODEL)
            model = AutoModelForCausalLM.from_pretrained(
                config.GENERATION_MODEL,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None
            )
            
            self.generator = pipeline(
                "text-generation",
                model=model,
                tokenizer=self.tokenizer,
                device=0 if self.device == "cuda" else -1,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

    async def process_document(self, file_path: str, filename: str) -> Dict:
        """Process a document and add it to the knowledge base"""
        try:
            logger.info(f"Processing document: {filename}")
            
            # Extract text
            text = self.text_extractor.extract_text_from_file(file_path)
            
            if not text.strip():
                raise ValueError("No text content found in the document")
            
            # Chunk text
            chunks_with_metadata = self.document_processor.chunk_text(text, filename)
            logger.info(f"Created {len(chunks_with_metadata)} chunks for {filename}")
            
            # Store document info
            doc_info = {
                'filename': filename,
                'file_path': file_path,
                'text': text,
                'chunks': [chunk.text for chunk in chunks_with_metadata],
                'upload_time': datetime.now().isoformat(),
                'chunk_count': len(chunks_with_metadata)
            }
            
            self.documents.append(doc_info)
            
            # Add chunks and metadata
            for chunk_info in chunks_with_metadata:
                self.document_chunks.append(chunk_info.text)
                self.chunk_metadata.append({
                    'filename': filename,
                    'chunk_id': chunk_info.chunk_id,
                    'char_start': chunk_info.char_start,
                    'char_end': chunk_info.char_end
                })
            
            # Update embeddings
            await self._update_embeddings()
            
            # Save data
            self._save_data()
            
            return {
                'status': 'success',
                'filename': filename,
                'chunk_count': len(chunks_with_metadata),
                'total_chunks': len(self.document_chunks)
            }
            
        except Exception as e:
            logger.error(f"Error processing document {filename}: {e}")
            raise

    async def _update_embeddings(self):
        """Update embeddings and FAISS index"""
        if not self.document_chunks:
            return
            
        logger.info("Generating embeddings...")
        
        # Generate embeddings
        self.embeddings = self.embedding_service.generate_embeddings(self.document_chunks)
        
        # Create FAISS index
        self.faiss_index = self.embedding_service.create_faiss_index(self.embeddings)
        
        # Save to disk
        self.storage.save_embeddings(self.embeddings)
        self.storage.save_faiss_index(self.faiss_index)
        
        logger.info(f"Updated FAISS index with {len(self.document_chunks)} chunks")

    def retrieve_relevant_chunks(self, query: str) -> List[RetrievalResult]:
        """Retrieve most relevant document chunks for a query"""
        if self.faiss_index is None or len(self.document_chunks) == 0:
            return []
        
        # Search for similar chunks
        scores, indices = self.embedding_service.search_similar(
            query, 
            self.faiss_index, 
            min(config.TOP_K_RETRIEVAL, len(self.document_chunks))
        )
        
        # Return chunks with scores and metadata
        results = []
        for score, idx in zip(scores, indices):
            if idx < len(self.document_chunks) and score > 0.3:
                results.append(RetrievalResult(
                    text=self.document_chunks[idx],
                    score=float(score),
                    metadata=self.chunk_metadata[idx] if idx < len(self.chunk_metadata) else {}
                ))
        
        return results

    async def generate_response(self, query: str) -> Dict:
        """Generate response based on retrieved documents"""
        start_time = datetime.now()
        
        try:
            self.initialize_generation_model()
            
            # Retrieve relevant chunks
            relevant_chunks = self.retrieve_relevant_chunks(query)
            
            if not relevant_chunks:
                return {
                    'response': "I couldn't find relevant information in the uploaded documents to answer your question.",
                    'confidence': 0.0,
                    'sources': [],
                    'processing_time': (datetime.now() - start_time).total_seconds()
                }
            
            # Prepare context
            context_parts = []
            sources = []
            
            for i, chunk_info in enumerate(relevant_chunks[:5]):
                context_parts.append(f"[Context {i+1}]: {chunk_info.text}")
                sources.append(f"{chunk_info.metadata.get('filename', 'Unknown')} (Score: {chunk_info.score:.3f})")
            
            context = "\n\n".join(context_parts)
            
            # Create prompt
            prompt = f"""Based on the following context from uploaded documents, provide a comprehensive and accurate answer to the question. If the context doesn't contain enough information, acknowledge the limitation.

Context:
{context}

Question: {query}

Instructions: Provide a detailed, well-structured answer based on the context. Be specific and cite relevant information when possible.

Answer:"""
            
            # Generate response
            response_output = self.generator(
                prompt,
                max_new_tokens=config.MAX_RESPONSE_LENGTH,
                temperature=config.TEMPERATURE,
                do_sample=True,
                top_p=0.9,
                top_k=50,
                repetition_penalty=1.1,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
            
            # Extract response
            generated_text = response_output[0]['generated_text']
            response = generated_text[len(prompt):].strip()
            response = re.sub(r'\n+', '\n', response).strip()
            
            # Calculate confidence
            weights = [chunk_info.score for chunk_info in relevant_chunks]
            confidence = np.average(weights, weights=weights) if weights else 0.0
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return {
                'response': response,
                'confidence': float(confidence),
                'sources': sources,
                'processing_time': processing_time
            }
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return {
                'response': f"I encountered an error while processing your question: {str(e)}",
                'confidence': 0.0,
                'sources': [],
                'processing_time': (datetime.now() - start_time).total_seconds()
            }

    def _save_data(self):
        """Save processed data"""
        self.storage.save_data(self.documents, self.document_chunks, self.chunk_metadata)

    def _load_existing_data(self):
        """Load existing processed data"""
        self.documents, self.document_chunks, self.chunk_metadata = self.storage.load_data()
        self.embeddings = self.storage.load_embeddings()
        self.faiss_index = self.storage.load_faiss_index()

    def get_document_info(self) -> List[DocumentInfo]:
        """Get information about uploaded documents"""
        return [
            DocumentInfo(
                filename=doc['filename'],
                upload_time=doc.get('upload_time', 'Unknown'),
                chunk_count=doc.get('chunk_count', 0),
                status='Processed'
            )
            for doc in self.documents
        ]

    def clear_all_data(self):
        """Clear all processed data"""
        self.documents = []
        self.document_chunks = []
        self.chunk_metadata = []
        self.embeddings = None
        self.faiss_index = None
        
        self.storage.clear_all_data()
        logger.info("All data cleared")
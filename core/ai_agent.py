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
        self.use_simple_generation = False
        
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
        """Initialize generation model with multiple fallback options"""
        if self.generator is None:
            logger.info("Loading generation model...")
            
            # Try different models in order of preference
            models_to_try = [
                "microsoft/DialoGPT-medium",  # Smaller than large
                "distilgpt2",                 # Very lightweight
                "gpt2"                        # Basic GPT-2
            ]
            
            for model_name in models_to_try:
                try:
                    logger.info(f"Attempting to load {model_name}...")
                    
                    self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                    
                    # Simple model loading without device_map complications
                    model = AutoModelForCausalLM.from_pretrained(model_name)
                    
                    # Create pipeline with minimal parameters
                    self.generator = pipeline(
                        "text-generation",
                        model=model,
                        tokenizer=self.tokenizer,
                        device=0 if self.device == "cuda" else -1
                    )
                    
                    # Set pad token
                    if self.tokenizer.pad_token is None:
                        self.tokenizer.pad_token = self.tokenizer.eos_token
                    
                    logger.info(f"Successfully loaded {model_name}")
                    break
                    
                except Exception as e:
                    logger.warning(f"Failed to load {model_name}: {e}")
                    continue
            
            if self.generator is None:
                logger.warning("All models failed to load, using simple text generation")
                self.use_simple_generation = True

    def _simple_text_generation(self, context: str, query: str) -> str:
        """Fallback simple text generation when models fail"""
        # Extract key sentences from context
        sentences = re.split(r'[.!?]+', context)
        relevant_sentences = [s.strip() for s in sentences if len(s.strip()) > 20][:3]
        
        if not relevant_sentences:
            return "I found relevant information but couldn't process it properly."
        
        # Create a simple response
        response = f"Based on your health insurance document, here's what I found:\n\n"
        for i, sentence in enumerate(relevant_sentences, 1):
            response += f"{i}. {sentence.strip()}.\n"
        
        response += f"\nThis information is related to your question: '{query}'"
        return response

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
            # Retrieve relevant chunks first
            relevant_chunks = self.retrieve_relevant_chunks(query)
            
            if not relevant_chunks:
                return {
                    'response': "I couldn't find relevant information in the uploaded documents to answer your question. Please try rephrasing your question or upload more relevant documents.",
                    'confidence': 0.0,
                    'sources': [],
                    'processing_time': (datetime.now() - start_time).total_seconds()
                }
            
            # Prepare context and sources
            context_parts = []
            sources = []
            
            for i, chunk_info in enumerate(relevant_chunks[:3]):  # Use fewer chunks for better processing
                context_parts.append(chunk_info.text)
                sources.append(f"{chunk_info.metadata.get('filename', 'Unknown')} (Score: {chunk_info.score:.3f})")
            
            context = "\n\n".join(context_parts)
            
            # Try to initialize and use the generation model
            try:
                self.initialize_generation_model()
                
                if self.use_simple_generation or self.generator is None:
                    # Use simple generation fallback
                    response = self._simple_text_generation(context, query)
                else:
                    # Use the actual generation model
                    prompt = f"""Context from health insurance document:
{context[:1000]}  

Question: {query}

Answer based on the context above:"""

                    try:
                        generated = self.generator(
                            prompt,
                            max_length=len(prompt) + 200,
                            max_new_tokens=150,
                            temperature=0.7,
                            do_sample=True,
                            pad_token_id=self.tokenizer.eos_token_id,
                            eos_token_id=self.tokenizer.eos_token_id,
                            truncation=True
                        )
                        
                        full_text = generated[0]['generated_text']
                        response = full_text[len(prompt):].strip()
                        
                        # Clean up the response
                        response = re.sub(r'\n+', '\n', response).strip()
                        
                        # If response is too short or empty, use fallback
                        if len(response.strip()) < 20:
                            response = self._simple_text_generation(context, query)
                            
                    except Exception as gen_error:
                        logger.error(f"Generation error: {gen_error}")
                        response = self._simple_text_generation(context, query)
                        
            except Exception as model_error:
                logger.error(f"Model initialization error: {model_error}")
                response = self._simple_text_generation(context, query)
            
            # Calculate confidence
            weights = [chunk_info.score for chunk_info in relevant_chunks]
            confidence = np.mean(weights) if weights else 0.0
            
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
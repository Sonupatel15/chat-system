from typing import List, Dict, Any
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

class EmbeddingGenerator:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize the embedding generator.
        
        Args:
            model_name (str): Name of the sentence-transformers model to use
        """
        self.model = SentenceTransformer(model_name)
        
    def generate_embeddings(self, documents: List[Dict]) -> List[Dict]:
        """Generate embeddings for a list of documents.
        
        Args:
            documents (List[Dict]): List of documents with 'text' and metadata
            
        Returns:
            List[Dict]: Documents enriched with embeddings
        """
        texts = [doc["text"] for doc in documents]
        
        # Generate embeddings with progress bar
        embeddings = []
        for text in tqdm(texts, desc="Generating embeddings"):
            embedding = self.model.encode(text)
            embeddings.append(embedding)
        
        # Add embeddings to documents
        for i, doc in enumerate(documents):
            doc["embedding"] = embeddings[i]
            
        return documents
    
    def generate_query_embedding(self, query: str) -> np.ndarray:
        """Generate embedding for a query string.
        
        Args:
            query (str): Query text
            
        Returns:
            np.ndarray: Query embedding
        """
        return self.model.encode(query)
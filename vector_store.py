import os
import json
import faiss
import numpy as np
from typing import List, Dict, Any, Tuple, Optional

class FAISSVectorStore:
    def __init__(self, store_dir: str = "vector_store"):
        """Initialize the FAISS vector store.
        
        Args:
            store_dir (str): Directory to store FAISS indices and metadata
        """
        self.store_dir = store_dir
        os.makedirs(store_dir, exist_ok=True)
        
        self.index_path = os.path.join(store_dir, "faiss_index.index")
        self.metadata_path = os.path.join(store_dir, "metadata.json")
        
        # Initialize index and metadata
        self.index = None
        self.metadata = []
        
        # Load existing index and metadata if they exist
        self.load_index_and_metadata()
    
    def load_index_and_metadata(self) -> None:
        """Load FAISS index and metadata from disk if they exist."""
        # Load metadata
        if os.path.exists(self.metadata_path):
            with open(self.metadata_path, 'r') as f:
                self.metadata = json.load(f)
        
        # Load FAISS index
        if os.path.exists(self.index_path):
            try:
                self.index = faiss.read_index(self.index_path)
            except Exception as e:
                print(f"Error loading FAISS index: {e}")
                # Initialize a new index if loading fails
                self.initialize_new_index(768)  # Default dimension for all-MiniLM-L6-v2
        
    def initialize_new_index(self, dimension: int) -> None:
        """Initialize a new FAISS index.
        
        Args:
            dimension (int): Embedding dimension
        """
        self.index = faiss.IndexFlatL2(dimension)
        
    def add_documents(self, documents: List[Dict]) -> List[str]:
        """Add documents to the vector store.
        
        Args:
            documents (List[Dict]): Documents with text, embeddings, and metadata
            
        Returns:
            List[str]: List of document IDs
        """
        # Extract embeddings and convert to numpy array
        embeddings = [doc["embedding"] for doc in documents]
        embeddings_array = np.array(embeddings).astype('float32')
        
        # Initialize index if it doesn't exist
        if self.index is None:
            dimension = embeddings_array.shape[1]
            self.initialize_new_index(dimension)
        
        # Add embeddings to index
        if len(embeddings) > 0:
            self.index.add(embeddings_array)
        
        # Update metadata
        doc_ids = []
        for i, doc in enumerate(documents):
            doc_id = doc.get("id", f"doc_{len(self.metadata)}")
            doc_ids.append(doc_id)
            
            # Store metadata without the embedding to save space
            metadata_entry = {
                "id": doc_id,
                "text": doc["text"],
                "metadata": doc.get("metadata", {})
            }
            self.metadata.append(metadata_entry)
        
        # Save index and metadata
        self.save()
        
        return doc_ids
    
    def similarity_search(self, query_embedding: np.ndarray, k: int = 5) -> List[Dict]:
        """Search for similar documents.
        
        Args:
            query_embedding (np.ndarray): Query embedding
            k (int): Number of results to return
            
        Returns:
            List[Dict]: List of similar documents with scores
        """
        if self.index is None or self.index.ntotal == 0:
            return []
        
        # Reshape for single query
        query_embedding = query_embedding.reshape(1, -1).astype('float32')
        
        # Limit k to the number of documents we have
        k = min(k, self.index.ntotal)
        
        # Perform search
        distances, indices = self.index.search(query_embedding, k)
        
        # Get results with metadata
        results = []
        for i, idx in enumerate(indices[0]):
            if idx != -1 and idx < len(self.metadata):  # Skip invalid indices
                doc = self.metadata[idx].copy()
                doc["score"] = float(distances[0][i])
                results.append(doc)
        
        return results
    
    def save(self) -> None:
        """Save index and metadata to disk."""
        if self.index is not None:
            faiss.write_index(self.index, self.index_path)
        
        with open(self.metadata_path, 'w') as f:
            json.dump(self.metadata, f)
    
    def get_document_count(self) -> int:
        """Get the number of documents in the index.
        
        Returns:
            int: Number of documents
        """
        if self.index is None:
            return 0
        return self.index.ntotal
    
    def get_source_documents(self) -> List[str]:
        """Get a list of unique source documents in the index.
        
        Returns:
            List[str]: List of source filenames
        """
        if not self.metadata:
            return []
        
        sources = set()
        for item in self.metadata:
            if "metadata" in item and "source" in item["metadata"]:
                sources.add(item["metadata"]["source"])
        
        return list(sources)
    
    def delete_document(self, source: str) -> None:
        """Delete all documents from a specific source file.
        
        Note: This is implemented by creating a new index without the specified documents.
        
        Args:
            source (str): Source filename to delete
        """
        if self.index is None or not self.metadata:
            return
        
        # Find all indices that don't belong to the source
        keep_indices = []
        new_metadata = []
        
        for i, item in enumerate(self.metadata):
            if item.get("metadata", {}).get("source") != source:
                keep_indices.append(i)
                new_metadata.append(item)
        
        if len(keep_indices) == len(self.metadata):
            # Nothing to delete
            return
        
        # Get embeddings to keep
        if len(keep_indices) > 0:
            # Create new index with same dimension
            dimension = self.index.d
            new_index = faiss.IndexFlatL2(dimension)
            
            # Add selected vectors to new index
            keep_indices = np.array(keep_indices)
            vectors = np.zeros((len(keep_indices), dimension), dtype=np.float32)
            
            for new_i, old_i in enumerate(keep_indices):
                # Extract vector at old_i
                vector = np.zeros((1, dimension), dtype=np.float32)
                self.index.reconstruct(old_i, vector[0])
                vectors[new_i] = vector
            
            # Add vectors to new index
            new_index.add(vectors)
            
            # Replace old index and metadata
            self.index = new_index
            self.metadata = new_metadata
            
            # Save changes
            self.save()
        else:
            # All documents were deleted, reset index
            self.index = None
            self.metadata = []
            self.save()
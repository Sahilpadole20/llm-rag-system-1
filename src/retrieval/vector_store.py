import os
import faiss
import numpy as np
import pickle
from typing import List, Any
from sentence_transformers import SentenceTransformer


class FaissVectorStore:
    """
    FAISS Vector Store following RAG-Tutorials pattern.
    Handles vector storage, indexing, and similarity search.
    """
    def __init__(self, persist_dir: str = "network_faiss_store", embedding_model: str = "all-MiniLM-L6-v2"):
        self.persist_dir = persist_dir
        os.makedirs(self.persist_dir, exist_ok=True)
        self.index = None
        self.metadata = []
        self.embedding_model = embedding_model
        self.model = SentenceTransformer(embedding_model)
        print(f"[INFO] Initialized vector store: {embedding_model}")

    def build_from_documents(self, documents: List[Any]):
        """Build vector store from documents using embedding pipeline"""
        print(f"[INFO] Building vector store from {len(documents)} raw documents...")
        
        # Use global embedding pipeline
        from embeddings.embedding_models import EmbeddingPipeline
        
        emb_pipe = EmbeddingPipeline()
        chunks = emb_pipe.chunk_documents(documents)
        embeddings = emb_pipe.embed_chunks(chunks)
        
        metadatas = []
        for chunk in chunks:
            # Combine content and metadata for better context
            metadata = {
                "text": chunk.page_content,
                **chunk.metadata  # Include original metadata
            }
            metadatas.append(metadata)
        
        self.add_embeddings(np.array(embeddings).astype('float32'), metadatas)
        self.save()
        print(f"[INFO] Vector store built and saved to {self.persist_dir}")

    def add_embeddings(self, embeddings: np.ndarray, metadatas: List[Any] = None):
        dim = embeddings.shape[1]
        if self.index is None:
            self.index = faiss.IndexFlatL2(dim)
        self.index.add(embeddings)
        if metadatas:
            self.metadata.extend(metadatas)
        print(f"[INFO] Added {embeddings.shape[0]} vectors to FAISS index.")

    def save(self):
        faiss_path = os.path.join(self.persist_dir, "faiss.index")
        meta_path = os.path.join(self.persist_dir, "metadata.pkl")
        faiss.write_index(self.index, faiss_path)
        with open(meta_path, "wb") as f:
            pickle.dump(self.metadata, f)
        print(f"[INFO] Saved FAISS index and metadata to {self.persist_dir}")

    def load(self):
        faiss_path = os.path.join(self.persist_dir, "faiss.index")
        meta_path = os.path.join(self.persist_dir, "metadata.pkl")
        if os.path.exists(faiss_path) and os.path.exists(meta_path):
            self.index = faiss.read_index(faiss_path)
            with open(meta_path, "rb") as f:
                self.metadata = pickle.load(f)
            print(f"[INFO] Loaded FAISS index and metadata from {self.persist_dir}")
            return True
        return False

    def search(self, query_embedding: np.ndarray, top_k: int = 5):
        if self.index is None:
            return []
        D, I = self.index.search(query_embedding, top_k)
        results = []
        for idx, dist in zip(I[0], D[0]):
            if idx >= 0 and idx < len(self.metadata):
                meta = self.metadata[idx]
                results.append({"index": idx, "distance": dist, "metadata": meta})
        return results

    def query(self, query_text: str, top_k: int = 5):
        """Query the vector store using a text input."""
        print(f"[INFO] Querying vector store for: '{query_text}'")
        query_emb = self.model.encode([query_text]).astype('float32')
        return self.search(query_emb, top_k=top_k)
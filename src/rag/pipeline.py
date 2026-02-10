from typing import List, Any
from src.retrieval.vector_store import FaissVectorStore
from src.llm.chat_models import ChatModel
from src.embeddings.embedding_models import EmbeddingModel

class RAGPipeline:
    def __init__(self, vector_store: FaissVectorStore, llm_model: ChatModel, embedding_model: EmbeddingModel):
        self.vector_store = vector_store
        self.llm_model = llm_model
        self.embedding_model = embedding_model

    def run(self, query: str, top_k: int = 5) -> str:
        # Step 1: Retrieve relevant documents
        query_embedding = self.embedding_model.embed(query)
        retrieved_docs = self.vector_store.query(query_embedding, top_k)

        # Step 2: Prepare context for LLM
        context = self.prepare_context(retrieved_docs)

        # Step 3: Generate response using LLM
        response = self.llm_model.generate_response(query, context)
        return response

    def prepare_context(self, retrieved_docs: List[Any]) -> str:
        context = "\n\n---\n\n".join([doc['text'] for doc in retrieved_docs])
        return context

    def optimize_query(self, query: str) -> str:
        # Placeholder for any query optimization logic
        return query.strip()
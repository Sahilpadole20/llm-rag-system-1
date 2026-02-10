from typing import List, Any
from langchain_groq import ChatGroq
from src.retrieval.vector_store import FaissVectorStore

class RetrievalQA:
    def __init__(self, vector_store: FaissVectorStore, llm_model: str):
        self.vector_store = vector_store
        self.llm = ChatGroq(model=llm_model)

    def answer_question(self, query: str, top_k: int = 5) -> str:
        """Retrieve relevant documents and generate an answer."""
        results = self.vector_store.query(query, top_k=top_k)
        
        # Extract text content from results
        context = "\n\n---\n\n".join([result["metadata"]["text"] for result in results if "metadata" in result])
        
        if not context:
            return "No relevant information found."
        
        prompt = f"Based on the following information, answer the question:\n\n{context}\n\nQuestion: {query}\nAnswer:"
        
        response = self.llm.invoke(prompt)
        return response.content.strip()
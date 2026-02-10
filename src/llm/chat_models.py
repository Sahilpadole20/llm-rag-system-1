import os
from langchain_groq import ChatGroq


class NetworkRAGSearch:
    """
    RAG Search following RAG-Tutorials pattern.
    Combines vector search with LLM for intelligent responses.
    """
    def __init__(self, vector_store, llm_model: str = "llama-3.3-70b-versatile"):
        self.vectorstore = vector_store
        
        # Initialize LLM with API key from environment
        groq_api_key = os.environ.get("GROQ_API_KEY")
        if not groq_api_key:
            raise ValueError("GROQ_API_KEY environment variable not set")
        self.llm = ChatGroq(groq_api_key=groq_api_key, model_name=llm_model)
        print(f"[INFO] Groq LLM initialized: {llm_model}")

    def search_and_summarize(self, query: str, top_k: int = 5) -> str:
        """Main RAG pipeline: retrieve relevant docs and generate response"""
        print(f"[INFO] Processing query: '{query}'")
        
        # Retrieve relevant documents
        results = self.vectorstore.query(query, top_k=top_k)
        
        # Extract text content
        texts = []
        for r in results:
            if r["metadata"] and "text" in r["metadata"]:
                texts.append(r["metadata"]["text"])
        
        context = "\n\n---\n\n".join(texts)
        
        if not context:
            return "No relevant network configurations found."
        
        # Create comprehensive prompt
        prompt = f"""You are a network performance expert. Analyze the following network configuration data to answer the user's question.

Query: {query}

Network Configuration Data:
{context}

Please provide a detailed analysis that includes:
1. Direct answer to the question
2. Relevant performance metrics and patterns
3. Specific configuration recommendations
4. Technical insights about network optimization

Answer:"""

        # Get LLM response
        response = self.llm.invoke(prompt)
        return response.content

    def find_optimal_configs(self, optimization_target: str, top_k: int = 3) -> str:
        """Find optimal configurations for specific targets"""
        queries = {
            "latency": "lowest latency high performance fast response edge deployment",
            "cost": "low cost budget efficient affordable deployment",
            "energy": "energy efficient low power consumption green deployment",
            "throughput": "high throughput data rate bandwidth performance"
        }
        
        query = queries.get(optimization_target.lower(), f"optimal {optimization_target} configuration")
        return self.search_and_summarize(query, top_k)

    def compare_layers(self) -> str:
        """Compare different deployment layers"""
        query = "compare edge cloud fog deployment layers performance latency cost energy"
        return self.search_and_summarize(query, top_k=10)
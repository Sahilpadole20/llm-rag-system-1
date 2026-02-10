class PromptTemplates:
    """
    A class to hold various prompt templates for querying the language model.
    """

    @staticmethod
    def basic_query_template(query: str) -> str:
        return f"Please provide a detailed response to the following query: {query}"

    @staticmethod
    def context_aware_template(query: str, context: str) -> str:
        return f"Given the context: {context}, please answer the following question: {query}"

    @staticmethod
    def optimization_template(target: str) -> str:
        return f"What are the best configurations for {target} optimization?"

    @staticmethod
    def comparison_template(layer1: str, layer2: str) -> str:
        return f"Compare the performance of {layer1} and {layer2} deployment layers."

    @staticmethod
    def detailed_analysis_template(query: str, context: str) -> str:
        return f"Analyze the following data and provide insights: {context}\n\nQuery: {query}"
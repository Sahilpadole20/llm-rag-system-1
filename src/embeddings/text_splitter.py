from typing import List

class RecursiveCharacterTextSplitter:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_text(self, text: str) -> List[str]:
        """
        Splits the input text into chunks of specified size with overlap.
        """
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunk = text[start:end]
            chunks.append(chunk)
            start += self.chunk_size - self.chunk_overlap
        return chunks

    def split_documents(self, documents: List[str]) -> List[str]:
        """
        Splits a list of documents into manageable chunks.
        """
        all_chunks = []
        for doc in documents:
            all_chunks.extend(self.split_text(doc))
        return all_chunks

class TextSplitter(RecursiveCharacterTextSplitter):
    """Alias for RecursiveCharacterTextSplitter for compatibility"""
    pass

def split_documents(documents: List[str], chunk_size: int = 1000, chunk_overlap: int = 200) -> List[str]:
    """
    Splits a list of documents into manageable chunks.
    """
    splitter = RecursiveCharacterTextSplitter(chunk_size, chunk_overlap)
    all_chunks = []
    for doc in documents:
        all_chunks.extend(splitter.split_text(doc))
    return all_chunks
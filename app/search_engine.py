from .rag import MumsRAG


class SearchEngine:
    def __init__(self):
        """Initialize the search engine with a RAG system."""
        self.rag = MumsRAG()

    def search(self, query: str) -> list[str]:
        """
        Search for relevant information in the documents.
        
        Args:
            query: The search query
            
        Returns:
            Relevant information or answers
        """
        return self.rag.query(query)

    def list_documents(self) -> list[str]:
        """
        List all processed documents in the vector store.
        
        Returns:
            List of documents processed by the RAG system
        """
        return self.rag.list_documents()

    def add_document(self, file_path: str) -> None:
        """
        Add a new document to the search engine.
        
        Args:
            file_path: Path to the document to add
        """
        self.rag.process_document(file_path)

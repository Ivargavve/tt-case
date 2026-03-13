from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings

# Load the OPENAI_API_KEY from the .env file
load_dotenv()

class MumsRAG:
    def __init__(self):
        """Initialize the RAG system with LLM, embeddings and vector store."""

        # NOTE: This is examples of tools you can use in the RAG system
        self.llm = ChatOpenAI(model="gpt-4o-mini")
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.vectorstore = FAISS(
            embedding_function=self.embeddings,
            index=None,
            docstore={},
            index_to_docstore_id={}
        )

    def list_documents(self) -> list[str]:
        """List all documents in the vector store."""
        # TODO: Implement this method
        pass

    def process_document(self, file_path: str) -> None:
        """Process a document and add it to the vector store."""
        # TODO: Implement this method
        pass

    def query(self, question: str) -> list[str]:
        """Query the RAG system with a question."""
        # TODO: Implement this method
        return [self.llm.invoke(question)] # NOTE: Ensure that you can invoke the llm, you can change this line when you implement the method.
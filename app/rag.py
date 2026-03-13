from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os

load_dotenv()

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


class MumsRAG:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o-mini")
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.vectorstore = None
        self.documents = []
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
        )

    def list_documents(self) -> list[str]:
        return self.documents

    def process_document(self, file_path: str) -> None:
        full_path = os.path.join(DATA_DIR, file_path)

        loader = PyPDFLoader(full_path)
        pages = loader.load()
        chunks = self.text_splitter.split_documents(pages)

        if self.vectorstore is None:
            self.vectorstore = FAISS.from_documents(chunks, self.embeddings)
        else:
            self.vectorstore.add_documents(chunks)

        self.documents.append(file_path)

    def query(self, question: str) -> list[str]:
        # TODO: Implement in next step
        pass


if __name__ == "__main__":
    rag = MumsRAG()

    for file in os.listdir(DATA_DIR):
        if file.endswith(".pdf"):
            print(f"Processing {file}...")
            rag.process_document(file)

    print(f"Done. Loaded {len(rag.documents)} documents.")

    # Quick verification
    results = rag.vectorstore.similarity_search("Who is the CEO?", k=2)
    print(f"\nTest query 'Who is the CEO?' returned {len(results)} chunks:")
    for r in results:
        print(f"  - {r.page_content[:100]}...")

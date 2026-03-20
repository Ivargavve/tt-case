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
        if not question.strip():
            raise ValueError("Query cannot be empty")

        if self.vectorstore is None:
            return []

        # Lower score = more similar in FAISS
        results = self.vectorstore.similarity_search_with_score(question, k=4)

        # Filter out irrelevant results (L2 distance: lower = more similar)
        threshold = 1.5
        relevant = [doc for doc, score in results if score < threshold]

        return [doc.page_content for doc in relevant]
    
    def formulate_answer(self, question: str) -> str:
        chunks = self.query(question)
        context = "\n".join(chunks)
        if not context:
            return "No relevant info found"
        return self.llm.invoke(f"Given this context:\n{context}\n\nAnswer: {question}").content

if __name__ == "__main__":
    rag = MumsRAG()

    for file in os.listdir(DATA_DIR):
        if file.endswith(".pdf"):
            print(f"Processing {file}...")
            rag.process_document(file)

    print(f"Done. Loaded {len(rag.documents)} documents.")

    # Quick verification
    results = rag.vectorstore.similarity_search_with_score("Who is the CEO?", k=2)
    answer = rag.formulate_answer("Who is the CEO?")
    print(f"\nTest query 'Who is the CEO?' returned {len(results)} chunks:")
    for doc, score in results:
        print(f"  [score: {score:.4f}] {doc.page_content[:100]}...")
    print(f"formulated answer: {answer}")
 
import os
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader

# Load environment variables
load_dotenv()

BASE_DIR = os.getenv("KB_DIR", "knowledge_base")
FAISS_DIR = os.getenv("FAISS_DIR", "career_faiss_index")

def load_documents(base_dir):
    docs = []
    for root, _, files in os.walk(base_dir):
        for file in files:
            if file.endswith(".txt"):
                path = os.path.join(root, file)
                loader = TextLoader(path, encoding="utf-8")
                docs.extend(loader.load())
    return docs

print("📂 Loading knowledge base files...")
documents = load_documents(BASE_DIR)

if not documents:
    print(f"❌ No documents found in '{BASE_DIR}'. Please add .txt files.")
    exit()

print(f"✅ Loaded {len(documents)} documents.")

# Split into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
docs = text_splitter.split_documents(documents)

# ✅ Use Gemini embeddings with API key from .env
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

print("⚡ Creating FAISS vector store...")
db = FAISS.from_documents(docs, embeddings)

db.save_local(FAISS_DIR)
print(f"✅ FAISS index saved in '{FAISS_DIR}'")

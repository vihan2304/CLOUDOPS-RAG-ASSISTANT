import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# -----------------------------
# STEP 1: Load PDFs
# -----------------------------

DOCS_PATH = "docs"

documents = []

for file in os.listdir(DOCS_PATH):
    if file.endswith(".pdf"):
        pdf_path = os.path.join(DOCS_PATH, file)

        print(f"Loading: {file}")

        loader = PyPDFLoader(pdf_path)
        docs = loader.load()

        documents.extend(docs)

print(f"\nTotal pages loaded: {len(documents)}")

# -----------------------------
# STEP 2: Split into chunks
# -----------------------------

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = text_splitter.split_documents(documents)

print(f"Total chunks created: {len(chunks)}")

# -----------------------------
# STEP 3: Create Embeddings
# -----------------------------

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# -----------------------------
# STEP 4: Create FAISS Vector DB
# -----------------------------

vectorstore = FAISS.from_documents(
    chunks,
    embedding_model
)

# -----------------------------
# STEP 5: Save Vector DB
# -----------------------------

vectorstore.save_local("vectorstore")

print("\nVector database created successfully!")

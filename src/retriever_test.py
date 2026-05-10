from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# -----------------------------
# Load embedding model
# -----------------------------

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# -----------------------------
# Load FAISS vector database
# -----------------------------

vectorstore = FAISS.load_local(
    "vectorstore",
    embedding_model,
    allow_dangerous_deserialization=True
)

# -----------------------------
# Ask Question
# -----------------------------

query = input("Ask a question: ")

# -----------------------------
# Retrieve relevant chunks
# -----------------------------

results = vectorstore.similarity_search(query, k=3)

# -----------------------------
# Print Results
# -----------------------------

print("\nTop Relevant Chunks:\n")

for i, doc in enumerate(results, 1):
    print(f"\nResult {i}:")
    print("-" * 50)
    print(doc.page_content)
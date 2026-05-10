from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate

# -----------------------------
# Load Embedding Model
# -----------------------------

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# -----------------------------
# Load FAISS Vector Database
# -----------------------------

vectorstore = FAISS.load_local(
    "vectorstore",
    embedding_model,
    allow_dangerous_deserialization=True
)

# -----------------------------
# Load Phi3 Mini Model
# -----------------------------

llm = OllamaLLM(model="phi3:mini")

# -----------------------------
# Prompt Template
# -----------------------------

template = """
You are CloudOps AI, a professional Cloud Infrastructure and DevOps assistant.

Use ONLY the provided context to answer the question.

Do NOT use outside knowledge.

If the answer is not found in the context, reply exactly:
""Sorry, I couldn't find relevant information about that in the uploaded documents.""

Context:
{context}

Question:
{question}

Answer:
"""

prompt = PromptTemplate(
    template=template,
    input_variables=["context", "question"]
)

# -----------------------------
# Chat Loop
# -----------------------------

print("\nCloudOps AI by Vihaan")
print("Type 'exit' to quit.\n")

while True:

    question = input("Ask: ")

    if question.lower() == "exit":
        break

    # Retrieve documents with similarity scores
    docs_and_scores = vectorstore.similarity_search_with_score(
        question,
        k=3
    )

    # Store relevant docs
    relevant_docs = []

    for doc, score in docs_and_scores:
        if score < 2.0:
            relevant_docs.append(doc)

    # If nothing relevant found
    if not relevant_docs:
        print("\nAnswer:\n")
        print("I could not find relevant information in the documents.")
        print("\n" + "="*80 + "\n")
        continue

    # Combine context
    context = "\n\n".join(
        [doc.page_content for doc in relevant_docs]
    )

    # Create final prompt
    final_prompt = prompt.format(
        context=context,
        question=question
    )

    # Generate response
    response = llm.invoke(final_prompt)

    print("\nAnswer:\n")
    print(response)

    print("\n" + "="*80 + "\n")
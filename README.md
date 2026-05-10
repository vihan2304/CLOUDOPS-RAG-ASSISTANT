# 🚀 CloudOps AI

## A Retrieval-Augmented Generation (RAG) Based DevOps Assistant using LangChain and Ollama

CloudOps AI is an enterprise-style RAG chatbot designed for Cloud Infrastructure and DevOps related question answering using custom documents.

The project combines semantic retrieval, vector databases, prompt engineering, and local LLM inference to generate grounded and context-aware responses while reducing hallucinations.

---

# ✨ Features

* 🔍 Retrieval-Augmented Generation (RAG)
* 🧠 Local LLM using Phi3 Mini via Ollama
* 📚 FAISS Vector Database
* 💬 Modern GPT-style Streamlit UI
* 🛡️ Hallucination Prevention System
* ⚡ Semantic Similarity Search
* 🕒 Sidebar Chat History
* 🎨 Professional Dark Theme Interface
* 🔒 Fully Offline AI Assistant

---

# 🛠️ Technology Stack

| Component              | Technology                   |
| ---------------------- | ---------------------------- |
| Programming Language   | Python                       |
| Frontend               | Streamlit                    |
| LLM Framework          | LangChain                    |
| Language Model Runtime | Ollama                       |
| LLM Model              | Phi3 Mini                    |
| Vector Database        | FAISS                        |
| Embedding Model        | HuggingFace all-MiniLM-L6-v2 |
| Semantic Retrieval     | Similarity Search            |

---

# 🏗️ System Architecture

```text
User Query
     ↓
Streamlit UI
     ↓
FAISS Retriever
     ↓
Relevant Chunks
     ↓
Prompt Engineering
     ↓
Phi3 Mini via Ollama
     ↓
Grounded AI Response
```

---

# 📂 Project Structure

```text
CLOUDOPS-RAG-ASSISTANT/
│
├── docs/
│   ├── cloud_fundamentals.pdf
│   ├── cloud_networking.pdf
│   ├── cloud_security.pdf
│   ├── docker_basics.pdf
│   └── kubernetes_basics.pdf
│
├── src/
│   ├── ingest.py
│   └── retriever_test.py
│
├── vectorstore/
│   ├── index.faiss
│   └── index.pkl
│
├── app.py
├── requirements.txt
├── README.md
└── CloudOps_AI_Project_Documentation.pdf
```

---

# 🚀 Installation & Setup

## 1️⃣ Clone Repository

```bash
https://github.com/vihan2304/CLOUDOPS-RAG-ASSISTANT.git
```

## 2️⃣ Create Virtual Environment

```bash
python -m venv venv
```

### Activate Environment

### Windows

```bash
venv\Scripts\activate
```

### Mac/Linux

```bash
source venv/bin/activate
```

## 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

## 4️⃣ Install Ollama

Download Ollama from:

https://ollama.com

## 5️⃣ Pull Phi3 Mini Model

```bash
ollama run phi3:mini
```

## 6️⃣ Run Application

```bash
streamlit run app.py
```

---

# 🛡️ Hallucination Prevention

The system uses multiple strategies to reduce hallucinated responses:

* Strict context-only prompting
* Similarity threshold filtering
* Fallback response handling
* Context validation checks

Fallback response used:

```text
Sorry, I couldn’t find relevant information about that in the uploaded documents.
```

---

# 📊 Current Features

✅ Semantic document retrieval
✅ Modern GPT-style UI
✅ Local LLM inference
✅ Session-based chat history
✅ Context-grounded responses
✅ FAISS vector search
✅ Streamlit interface

---

# ⚠️ Current Limitations

* No reranking pipeline
* No source citations
* Limited conversational memory
* No hybrid retrieval
* Limited multi-document support

---

# 🔮 Future Improvements

* Conversational memory
* Source citations
* Hybrid retrieval
* Streaming responses
* Multi-document support
* Cloud deployment
* PDF upload interface

---

# 📄 Documentation

Detailed project documentation is available in:

```text
Project_Documentation.pdf
```

---

# 👨‍💻 Developed By

## Vihaan Agarwal

---

# ⭐ If you like this project

Consider giving it a star on GitHub!

import streamlit as st
import time
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate

# -------------------------------------------------
# 1. PAGE CONFIG & SESSION STATE
# -------------------------------------------------
st.set_page_config(
    page_title="CloudOps AI by Vihaan",
    page_icon="🚀",
    layout="wide"
)

if "chat_sessions" not in st.session_state:
    st.session_state.chat_sessions = {"Default Chat": []}
if "active_session" not in st.session_state:
    st.session_state.active_session = "Default Chat"

messages = st.session_state.chat_sessions[st.session_state.active_session]

# -------------------------------------------------
# 2. CONSOLIDATED PREMIUM CSS (FIXED TOGGLE)
# -------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600&family=Plus+Jakarta+Sans:wght@700;800&display=swap');

    div[data-testid="stStatusWidget"] { display: none !important; }
    
    /* Keep the header logic but hide the buttons so the toggle stays functional */
    header[data-testid="stHeader"] {
        background: rgba(0,0,0,0) !important;
    }
    [data-testid="stHeaderActionElements"], #MainMenu {
        visibility: hidden !important;
    }

    .stApp {
        background: radial-gradient(circle at top right, #1e293b, #0b1020 70%);
        background-attachment: fixed;
        color: white;
        font-family: 'Inter', sans-serif !important;
    }

    .main-title {
        font-size: 3.8rem;
        font-weight: 800;
        letter-spacing: -2px;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        background: linear-gradient(90deg, #ffffff, #60a5fa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }

    .subtitle {
        color: #9ca3af;
        font-size: 1.1rem;
        margin-top: -10px;
        margin-bottom: 40px;
    }

    .chat-card {
        background: rgba(17, 24, 39, 0.6);
        backdrop-filter: blur(12px);
        padding: 25px;
        border-radius: 20px;
        margin-top: 10px;
        margin-bottom: 10px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        width: fit-content;
        max-width: 85%;
    }

    .user-question {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: 700;
        color: #60a5fa;
        margin-bottom: 10px;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
    }

    .answer-text { font-size: 1rem; line-height: 1.6; color: #f3f4f6; }

    .capability-pill {
        display: inline-block;
        padding: 6px 16px;
        border-radius: 50px;
        background: rgba(96, 165, 250, 0.1);
        border: 1px solid rgba(96, 165, 250, 0.2);
        font-size: 0.8rem;
        color: #9ca3af;
        margin-right: 8px;
        margin-bottom: 12px;
    }

    @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
    @keyframes fadeIn { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
    
    .thinking-spinner {
        border: 2px solid rgba(255, 255, 255, 0.1);
        border-top: 2px solid #60a5fa;
        border-radius: 50%;
        width: 16px; height: 16px;
        animation: spin 1s linear infinite;
        display: inline-block; margin-right: 10px; vertical-align: middle;
    }
    .thinking-text { color: #9ca3af; font-size: 0.9rem; padding: 10px 0; margin-bottom: 20px; animation: fadeIn 0.3s; }

    section[data-testid="stSidebar"] {
        background-color: rgba(10, 15, 28, 0.95);
        border-right: 1px solid rgba(96, 165, 250, 0.1);
    }

    .badge {
        background: rgba(96, 165, 250, 0.1);
        color: #60a5fa;
        padding: 4px 12px;
        border-radius: 12px;
        border: 1px solid rgba(96, 165, 250, 0.2);
        font-size: 0.8rem;
        display: inline-block;
        margin: 4px;
    }

    div[data-testid="stChatInput"] {
        max-width: 850px !important;
        margin: 0 auto !important;
        background: rgba(17, 24, 39, 0.4) !important;
        backdrop-filter: blur(10px) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 15px !important;
    }

    div[data-testid="stChatInput"]:after {
        content: "CloudOps is AI and can make mistakes.";
        display: block;
        text-align: center;
        font-size: 0.75rem;
        color: #4b5563;
        padding-top: 12px;
        padding-bottom: 20px;
    }

    div.stButton > button {
        background-color: rgba(255, 255, 255, 0.03) !important;
        color: white !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 15px !important;
        transition: all 0.3s ease !important;
        height: 80px !important;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
    }

    div.stButton > button:hover {
        background-color: rgba(96, 165, 250, 0.1) !important;
        border-color: #60a5fa !important;
        transform: translateY(-2px) !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# -------------------------------------------------
# 3. SIDEBAR (HISTORY & NEW CHAT)
# -------------------------------------------------
with st.sidebar:
    st.markdown("## 🚀 CloudOps AI")
    st.caption("Developed by Vihaan Agarwal")
    
    if st.button("New Chat", use_container_width=True):
        new_name = f"Chat {len(st.session_state.chat_sessions) + 1}"
        st.session_state.chat_sessions[new_name] = []
        st.session_state.active_session = new_name
        st.rerun()

    st.divider()
    st.markdown("### 🕒 History")
    for session_name in reversed(list(st.session_state.chat_sessions.keys())):
        is_active = session_name == st.session_state.active_session
        display_name = f"💬 {session_name[:20]}..." if len(session_name) > 20 else f"💬 {session_name}"
        if st.button(display_name, key=f"sess_{session_name}", use_container_width=True, 
                     type="primary" if is_active else "secondary"):
            st.session_state.active_session = session_name
            st.rerun()

    st.divider()
    st.markdown("### 📚 Knowledge Base")
    topics = ["Cloud", "Docker", "K8s", "Security"]
    st.markdown(" ".join([f'<span class="badge">{t}</span>' for t in topics]), unsafe_allow_html=True)

# -------------------------------------------------
# 4. ASSET LOADING
# -------------------------------------------------
@st.cache_resource
def load_assets():
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.load_local("vectorstore", embedding_model, allow_dangerous_deserialization=True)
    llm = OllamaLLM(model="phi3:mini")
    return vectorstore, llm

vectorstore, llm = load_assets()

# -------------------------------------------------
# 5. RESTORED WELCOME UI (ALL 4 CARDS)
# -------------------------------------------------
if not messages:
    st.markdown('<div class="main-title">🚀 CloudOps AI</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Professional Cloud Infrastructure & DevOps Assistant</div>', unsafe_allow_html=True)

    st.markdown("""
        <div style="margin-bottom: 25px;">
            <span class="capability-pill">⚡ RAG Enhanced</span>
            <span class="capability-pill">🔒 Private LLM</span>
            <span class="capability-pill">🐳 Docker & K8s</span>
        </div>
    """, unsafe_allow_html=True)
    
    # Restored 2x2 Grid
    col1, col2 = st.columns(2)
    col3, col4 = st.columns(2)
    
    actions = [
        {"icon": "☁️", "label": "Deploy K8s Cluster", "prompt": "How do I deploy a production-ready Kubernetes cluster?"},
        {"icon": "🔐", "label": "Zero Trust Security", "prompt": "What are the core principles of Zero Trust Security?"},
        {"icon": "🐳", "label": "Debug Container", "prompt": "How do I debug a failing Docker container?"},
        {"icon": "📊", "label": "Optimize Azure Costs", "prompt": "How can I optimize Azure infrastructure costs?"}
    ]
    
    for i, action in enumerate(actions):
        target = [col1, col2, col3, col4][i]
        with target:
            if st.button(f"{action['icon']} {action['label']}", key=f"act_{i}", use_container_width=True):
                st.session_state.temp_question = action['prompt']
                st.rerun()
else:
    st.markdown('<div style="height: 10px;"></div>', unsafe_allow_html=True) 

# -------------------------------------------------
# 6. CHAT RENDERER
# -------------------------------------------------
for msg in messages:
    is_user = msg["role"] == "user"
    align = "flex-end" if is_user else "flex-start"
    bg = "linear-gradient(135deg, #2563eb, #1d4ed8)" if is_user else "rgba(17, 24, 39, 0.7)"
    st.markdown(f'''
        <div style="display: flex; justify-content: {align}; animation: fadeIn 0.3s ease-out;">
            <div class="chat-card" style="background: {bg};">
                <div class="user-question">{"🧑 YOU" if is_user else "🤖 CLOUDOPS AI"}</div>
                <div class="answer-text">{msg["content"]}</div>
            </div>
        </div>
    ''', unsafe_allow_html=True)

# -------------------------------------------------
# 7. INPUT & PROCESSING
# -------------------------------------------------
active_q = st.chat_input("Ask CloudOps...")

if "temp_question" in st.session_state:
    active_q = st.session_state.temp_question
    del st.session_state.temp_question

if active_q:
    if not messages:
        old_name = st.session_state.active_session
        new_name = active_q[:25] + "..." if len(active_q) > 25 else active_q
        st.session_state.chat_sessions[new_name] = st.session_state.chat_sessions.pop(old_name)
        st.session_state.active_session = new_name
        messages = st.session_state.chat_sessions[new_name]
    messages.append({"role": "user", "content": active_q})
    st.rerun()

if messages and messages[-1]["role"] == "user":
    last_query = messages[-1]["content"]
    status = st.empty()
    status.markdown('<div class="thinking-text"><div class="thinking-spinner"></div>Thinking...</div>', unsafe_allow_html=True)

    docs_and_scores = vectorstore.similarity_search_with_score(last_query, k=3)
    relevant = [doc for doc, score in docs_and_scores if score < 1.2]
    context = "\n\n".join([d.page_content for d in relevant]) if relevant else "N/A"

    if relevant:
        template = "Context: {context}\n\nQuestion: {question}\nAnswer:"
        prompt = PromptTemplate(template=template, input_variables=["context", "question"])
        answer = llm.invoke(prompt.format(context=context, question=last_query))
    else:
        answer = "I couldn't find specific documentation for that. Please ask about DevOps or Cloud infrastructure."

    status.empty()
    resp_placeholder = st.empty()
    full_resp = ""
    for word in answer.split():
        full_resp += word + " "
        resp_placeholder.markdown(f'''
            <div style="display: flex; justify-content: flex-start;">
                <div class="chat-card" style="background: rgba(17, 24, 39, 0.7);">
                    <div class="user-question">🤖 CLOUDOPS AI</div>
                    <div class="answer-text">{full_resp}▌</div>
                </div>
            </div>
        ''', unsafe_allow_html=True)
        time.sleep(0.01)
    messages.append({"role": "assistant", "content": full_resp.strip()})
    st.rerun()
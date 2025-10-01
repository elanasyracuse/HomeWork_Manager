# HW5.py — Enhanced iSchool Student Orgs Chatbot

import os, glob
import streamlit as st

# --- Model SDK ---
from openai import OpenAI

# --- SQLite shim (for Streamlit Cloud + Chroma) ---
try:
    import pysqlite3, sys
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except Exception:
    pass
# --------------------------------------------------

# Core deps
import chromadb
from chromadb.config import Settings
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

# HTML parsing
from bs4 import BeautifulSoup

# ================================
# Page setup & API Keys
# ================================
st.set_page_config(page_title="Enhanced iSchool Chatbot (HW5)", page_icon="✨", layout="centered")
st.title("✨ Enhanced iSchool Student Orgs Chatbot")

# Load OpenAI API key from secrets.toml
OPENAI_KEY = st.secrets.get("OPENAI_API_KEY", "")

# ================================
# Config
# ================================
PERSIST_DIR = ".chromadb"
COLLECTION_NAME = "iSchoolOrgsCollection"
HTML_GLOB = "su_orgs/**/*.html"
EMBED_MODEL = "text-embedding-3-small"
K_CHUNKS = 5
TEMPERATURE = 0.2

# ================================
# Helper Functions (Backend)
# ================================

def read_html_text(path: str) -> str:
    try:
        with open(path, "rb") as f:
            soup = BeautifulSoup(f, "html.parser")
            for tag in soup(["script", "style", "noscript"]): tag.extract()
            text = soup.get_text(separator="\n")
            lines = [ln.strip() for ln in text.splitlines()]
            return "\n".join([ln for ln in lines if ln]).strip()
    except Exception as e:
        print(f"Failed to read HTML {path}: {e}"); return ""

def chunk_into_two_semantic(text: str):
    if not text: return ["", ""]
    paras = [p.strip() for p in text.split("\n") if p.strip()]
    if len(paras) >= 4:
        total = sum(len(p) for p in paras)
        target, s, best_idx, best_diff = total / 2, 0, None, 1e18
        for i, p in enumerate(paras[:-1]):
            s += len(p); diff = abs(s - target)
            if diff < best_diff: best_diff, best_idx = diff, i
        return ["\n".join(paras[:best_idx+1]).strip(), "\n".join(paras[best_idx+1:]).strip()]
    mid = max(1, len(text) // 2)
    return [text[:mid].strip(), text[mid:].strip()]

@st.cache_resource
def get_chroma_collection():
    """Cached function to get or build the ChromaDB collection."""
    client = chromadb.Client(Settings(persist_directory=PERSIST_DIR))
    
    if not OPENAI_KEY:
        st.error("OpenAI API key is required for the vector database's embedding function."); st.stop()
    embedder = OpenAIEmbeddingFunction(api_key=OPENAI_KEY, model_name=EMBED_MODEL)
    coll = client.get_or_create_collection(name=COLLECTION_NAME, embedding_function=embedder)

    if coll.count() == 0:
        html_paths = sorted(glob.glob(HTML_GLOB, recursive=True))
        if not html_paths:
            st.error(f"No HTML files found. Please check the path: {HTML_GLOB}"); st.stop()
        
        with st.spinner("Setting up the chatbot for the first time..."):
            ids, docs, metas = [], [], []
            for path in html_paths:
                fname = os.path.basename(path)
                text = read_html_text(path)
                if not text: continue
                c1, c2 = chunk_into_two_semantic(text)
                for i, ch in enumerate([c1, c2]):
                    ids.append(f"{fname}::chunk-{i}"); docs.append(ch)
                    metas.append({"source": fname, "chunk": i, "path": path})
            
            if ids:
                coll.add(ids=ids, documents=docs, metadatas=metas)
    return coll

# --- HW5 REFACTOR ---
# The original logic is now split into two distinct functions
# as per the assignment instructions.

def get_relevant_club_info(query: str):
    """
    HW5 Function 3a: Takes a query and returns relevant_club_info from the vector search.
    """
    coll = get_chroma_collection()
    res = coll.query(query_texts=[query], n_results=K_CHUNKS)
    
    # This is the relevant_club_info text from the vector search
    relevant_club_info = "\n\n".join(res.get("documents", [[]])[0])
    sources = list(set([m.get("source") for m in res.get("metadatas", [[]])[0] if m.get("source")]))
    
    return relevant_club_info, sources

def generate_llm_answer(query: str, context: str):
    """
    HW5 Function 3b: Invokes the LLM with the results of the vector search.
    """
    system_prompt = (
        "You are a helpful assistant for iSchool student organizations. "
        "Answer the user's question by synthesizing information from the provided context. "
        "If the context is insufficient to fully answer, provide what information you can and "
        "then state what you cannot answer. If the question is completely unrelated to the context, "
        "politely state that you can only answer questions about iSchool student orgs."
    )
    user_content = f"User question: {query}\n\nRetrieved context:\n{context}"
    
    if not OPENAI_KEY:
        st.error("OpenAI API key is missing!")
        return "Sorry, I encountered an error."
    
    client = OpenAI(api_key=OPENAI_KEY)
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=TEMPERATURE,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ]
    )
    answer = completion.choices[0].message.content
    return answer

# ================================
# Main Chat UI
# ================================
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for role, msg in st.session_state.chat_history:
    with st.chat_message(role):
        st.write(msg)

if user_q := st.chat_input("Ask about iSchool student organizations…"):
    st.session_state.chat_history.append(("user", user_q))
    with st.chat_message("user"): st.write(user_q)

    with st.chat_message("assistant"):
        with st.spinner("Searching for info and asking OpenAI..."):
            try:
                # HW5 Step 1: Get relevant info from the vector search
                relevant_info, sources = get_relevant_club_info(user_q)
                
                # HW5 Step 2: Pass that info to the LLM to generate an answer
                answer = generate_llm_answer(query=user_q, context=relevant_info)
                
                msg = answer
                if sources:
                    msg += "\n\n*Sources:* " + ", ".join(sources)
            except Exception as e:
                msg = f"An error occurred: {e}"
            st.write(msg)
            st.session_state.chat_history.append(("assistant", msg))

    # Trim memory to last 5 exchanges
    if len(st.session_state.chat_history) > 10:
        st.session_state.chat_history = st.session_state.chat_history[-10:]
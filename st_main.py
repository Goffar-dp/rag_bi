import os
import streamlit as st
import pandas as pd
import yaml
from ragbi import RAGSystem

# --- Streamlit Config ---
st.set_page_config(page_title="Bilingual RAG Chatbot", page_icon="🤖")
st.title("🌐 Bilingual RAG: Ask Your Documents")

# Initialize session state
if "rag" not in st.session_state:
    st.session_state.rag = RAGSystem()
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- Upload Files ---
uploaded_files = st.sidebar.file_uploader(
    "Upload Documents", 
    type=["pdf", "docx", "txt"],
    accept_multiple_files=True
)

# Process new files
if uploaded_files:
    for file in uploaded_files:
        file_bytes = file.getvalue()
        file_name = file.name
        chunks_added = st.session_state.rag.process_upload(file_bytes, file_name)
        if chunks_added:
            st.sidebar.success(f"Processed {chunks_added} chunks from {file_name}")

# --- Chat UI ---
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg:
            st.subheader("Sources:")
            st.dataframe(msg["sources"])

query = st.chat_input("Ask in English or Bangla")

if query:
    with st.chat_message("user"):
        st.markdown(query)
    
    with st.chat_message("assistant"):
        with st.spinner("Generating response..."):
            answer, docs = st.session_state.rag.get_response(query)
            st.markdown(answer)
            
            if docs:
                sources = st.session_state.rag.format_sources(docs)
                sources_df = pd.DataFrame(sources)
                st.subheader("Sources:")
                st.dataframe(sources_df)
            
            # Update history
            st.session_state.chat_history.append({
                "role": "user", 
                "content": query
            })
            st.session_state.chat_history.append({
                "role": "assistant", 
                "content": answer,
                "sources": sources_df if docs else None
            })

# Display processed files
if st.session_state.rag.processed_files:
    st.sidebar.subheader("Processed Files")
    for file_name in st.session_state.rag.processed_files.values():
        st.sidebar.write(f"✓ {file_name}")

import streamlit as st
import os
import json
from datetime import datetime
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import HuggingFaceHub
from langchain.chains import RetrievalQA

# --- Configuration ---
LLM_MODEL = "meta-llama/Llama-3.2-8B-Instruct"  # Open-access variant
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHAT_HISTORY_FILE = "chat_history.json"
PDF_FOLDER_PATH = "./hr_documents"  # Folder containing all HR-related PDFs

st.set_page_config(page_title="Auxilium Corp HRBP", layout="wide")

st.markdown("""
    <h1 style='text-align: center; color: #4CAF50;'>🤖 Auxilium Corp HRBP</h1>
    <h4 style='text-align: center;'>Your AI-Powered HR Assistant</h4>
    <hr style='border:1px solid #4CAF50'>
    """, unsafe_allow_html=True)

# --- User Login ---
st.sidebar.header("🔐 User Login")
username = st.sidebar.text_input("Enter your username")
if not username:
    st.warning("Please enter your username to continue.")
    st.stop()

# --- Load or initialize chat history ---
def load_chat_history():
    if os.path.exists(CHAT_HISTORY_FILE):
        with open(CHAT_HISTORY_FILE, "r") as f:
            return json.load(f)
    return {}

def save_chat_history(history):
    with open(CHAT_HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=4)

chat_history = load_chat_history()
user_history = chat_history.get(username, [])

# --- Load PDF documents from folder ---
@st.cache_resource(show_spinner=False)
def process_pdfs_from_folder(folder_path):
    documents = []
    try:
        for file_name in os.listdir(folder_path):
            if file_name.endswith(".pdf"):
                file_path = os.path.join(folder_path, file_name)
                loader = PyPDFLoader(file_path)
                documents.extend(loader.load())

        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        texts = splitter.split_documents(documents)

        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        vectordb = FAISS.from_documents(texts, embeddings)
        return vectordb
    except Exception as e:
        st.error(f"Error loading documents: {str(e)}")
        return None

with st.spinner("Loading HR documents and building vector store..."):
    vectordb = process_pdfs_from_folder(PDF_FOLDER_PATH)

# --- Set up LLM and RetrievalQA chain ---
@st.cache_resource(show_spinner=False)
def setup_qa_chain(vectordb):
    llm = HuggingFaceHub(repo_id=LLM_MODEL, model_kwargs={"temperature": 0.3, "max_new_tokens": 500})
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=vectordb.as_retriever(), chain_type="stuff")
    return qa

if vectordb:
    qa_chain = setup_qa_chain(vectordb)

    st.markdown("""
    <div style='background-color: #f0f0f0; padding: 15px; border-radius: 10px;'>
        <p><strong>Ask me anything about onboarding, policies, or our organizational culture!</strong></p>
    </div>
    """, unsafe_allow_html=True)

    user_query = st.text_input("💬 Your question:", placeholder="e.g. What is the leave policy at Auxilium?", key="user_query")

    if user_query:
        with st.spinner("Thinking..."):
            try:
                response = qa_chain.run(user_query)
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                user_history.append({"timestamp": timestamp, "query": user_query, "response": response})
                chat_history[username] = user_history
                save_chat_history(chat_history)

                st.success("Answer:")
                st.markdown(f"<div style='background-color: #e8f5e9; padding: 15px; border-radius: 10px;'>{response}</div>", unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Failed to generate response: {str(e)}")

    # --- Show chat history ---
    with st.expander("🕘 View Chat History"):
        if user_history:
            for entry in reversed(user_history):
                st.markdown(f"**[{entry['timestamp']}]** {entry['query']}\n- {entry['response']}")
        else:
            st.info("No chat history found.")
else:
    st.error("Failed to load knowledge base from the document folder.")

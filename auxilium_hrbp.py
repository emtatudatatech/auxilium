import streamlit as st
import google.generativeai as genai
import os
import fitz # PyMuPDF
import numpy as np
from dotenv import load_dotenv
import time
import logging

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables (especially Google API Key)
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    st.error("🚨 Google API Key not found! Please set the GOOGLE_API_KEY environment variable.")
    st.stop()

genai.configure(api_key=API_KEY)

# --- Constants ---
PDF_FOLDER_PATH = "hr_docs" # Containing the HR documents
EMBEDDING_MODEL = "models/text-embedding-004"
GENERATIVE_MODEL = "models/gemini-1.5-pro-latest"
SYSTEM_PROMPT = """
You are Auxilium Corp HRBP, a helpful AI assistant for Auxilium Corporation employees.
Your primary function is to answer questions based *only* on the provided context documents.
Follow these instructions strictly:
1.  Analyze the user's question.
2.  Review the provided document excerpts (context).
3.  Formulate a concise answer using *only* the information present in the context.
4.  Do *not* add any information that is not explicitly stated in the context.
5.  Do *not* mention your knowledge cutoff date or that you are an AI model unless asked.
6.  If the context does not contain the answer to the question, state clearly: "Based on the provided documents, I cannot answer that question." Do not make up information.
7.  Cite the source document(s) accurately at the end of your response, like this: *Source: [filename.pdf]*. If multiple documents were used, list them: *Sources: [file1.pdf, file2.pdf]*.
8.  Respond directly to the question. Do not start with phrases like "Based on the provided context..." unless it's necessary for clarity.
"""

# --- Caching ---
# Cache data loading and embedding generation to avoid recomputing on every interaction
# @st.cache_resource(show_spinner="Loading and processing HR documents...")
@st.cache_resource(show_spinner="Getting started...")
def load_and_embed_pdfs(folder_path):
    """Loads PDFs, chunks text, generates embeddings, and stores them."""
    # documents = []
    all_chunks_with_sources = []

    if not os.path.isdir(folder_path):
        st.error(f"❌ Error: Folder not found at {folder_path}")
        return None

    pdf_files = [f for f in os.listdir(folder_path) if f.lower().endswith(".pdf")]
    if not pdf_files:
        st.warning(f"⚠️ No PDF files found in the '{folder_path}' folder.")
        return None

    logging.info(f"Found {len(pdf_files)} PDF files in '{folder_path}'.")

    for filename in pdf_files:
        filepath = os.path.join(folder_path, filename)
        try:
            doc = fitz.open(filepath)
            logging.info(f"Processing '{filename}'...")
            full_text = ""
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                full_text += page.get_text("text") + "\n\n" # Add space between pages
            doc.close()

            # Simple chunking strategy (by paragraphs, adjust as needed)
            chunks = [chunk.strip() for chunk in full_text.split('\n\n') if chunk.strip()]

            if not chunks:
                logging.warning(f"No text chunks extracted from '{filename}'. Skipping.")
                continue

            logging.info(f"Extracted {len(chunks)} chunks from '{filename}'.")
            for chunk in chunks:
                all_chunks_with_sources.append({"text": chunk, "source": filename})

        except Exception as e:
            logging.error(f"Error processing PDF '{filename}': {e}")
            st.error(f"❌ Failed to process '{filename}'. Error: {e}")
            continue # Skip this file

    if not all_chunks_with_sources:
        st.error("❌ No text could be extracted from any PDF files.")
        return None

    logging.info(f"Total chunks extracted: {len(all_chunks_with_sources)}. Generating embeddings...")

    # Generate embeddings in batches (adjust batch_size based on API limits/performance)
    batch_size = 100 # Gemini API can handle up to 100 embeddings per request
    all_embeddings = []
    all_texts = [item['text'] for item in all_chunks_with_sources]
    all_sources = [item['source'] for item in all_chunks_with_sources]

    for i in range(0, len(all_texts), batch_size):
        batch_texts = all_texts[i:i+batch_size]
        try:
            # Introduce a small delay to avoid hitting rate limits aggressively
            time.sleep(1)
            result = genai.embed_content(
                model=EMBEDDING_MODEL,
                content=batch_texts,
                task_type="retrieval_document" # Specify task type for better embeddings
            )
            all_embeddings.extend(result['embedding'])
            logging.info(f"Generated embeddings for batch {i//batch_size + 1}...")
        except Exception as e:
            logging.error(f"Error generating embeddings for batch starting at index {i}: {e}")
            st.error(f"🚨 API Error during embedding generation: {e}. Check API Key and Quotas.")
            # Fill with None or handle differently if partial results are okay
            all_embeddings.extend([None] * len(batch_texts)) # Mark failed embeddings

    # Combine texts, sources, and embeddings, filtering out failed ones
    processed_data = []
    for text, source, embedding in zip(all_texts, all_sources, all_embeddings):
        if embedding is not None:
            processed_data.append({"text": text, "source": source, "embedding": np.array(embedding)})
        else:
            logging.warning(f"Skipping chunk from '{source}' due to embedding error.")

    if not processed_data:
         st.error("❌ Critical Error: Failed to generate any embeddings for the documents.")
         return None

    logging.info(f"Successfully generated embeddings for {len(processed_data)} chunks.")
    return processed_data

# --- Helper Functions ---
def find_best_matches(query, data, top_n=5):
    """Finds the most relevant document chunks based on cosine similarity."""
    if not data:
        return [], [] # Return empty lists if no data

    query_embedding_response = genai.embed_content(
        model=EMBEDDING_MODEL,
        content=query,
        task_type="retrieval_query" # Specify task type
    )
    query_embedding = np.array(query_embedding_response['embedding'])

    similarities = []
    doc_embeddings = np.array([item['embedding'] for item in data])

    # Calculate Cosine Similarity
    # Ensure embeddings are 1D arrays before dot product
    if query_embedding.ndim > 1: query_embedding = query_embedding.flatten()

    for i in range(doc_embeddings.shape[0]):
        doc_emb = doc_embeddings[i]
        if doc_emb.ndim > 1: doc_emb = doc_emb.flatten() # Ensure doc embedding is 1D

        # Basic check for zero vectors to avoid division by zero
        if np.linalg.norm(query_embedding) == 0 or np.linalg.norm(doc_emb) == 0:
            similarity = 0.0
        else:
            similarity = np.dot(query_embedding, doc_emb) / (np.linalg.norm(query_embedding) * np.linalg.norm(doc_emb))
        similarities.append(similarity)

    # Get indices of top_n similarities
    sorted_indices = np.argsort(similarities)[::-1]
    top_indices = sorted_indices[:top_n]

    # Retrieve the corresponding chunks and their sources
    best_matches_text = [data[i]['text'] for i in top_indices]
    best_matches_sources = list(set(data[i]['source'] for i in top_indices)) # Unique sources

    logging.info(f"Found {len(best_matches_text)} relevant chunks from sources: {best_matches_sources}")
    return best_matches_text, best_matches_sources

def get_gemini_response(query, context_chunks, sources):
    """Generates a response using Gemini based on the query and context."""
    if not context_chunks:
        return "Based on the current knowledgebase, I cannot answer that question. Please seek assistance from your HRBP.", []

    context_string = "\n\n---\n\n".join(context_chunks)

    # Prepare the messages for the chat model
    messages = [
        {'role': 'user',
         'parts': [SYSTEM_PROMPT + f"\n\nContext from documents:\n{context_string}\n\nQuestion:\n{query}"]}
    ]

    try:
        model = genai.GenerativeModel(GENERATIVE_MODEL)
        response = model.generate_content(messages) # Use generate_content for single-turn adjusted for system prompt

        # Basic check for empty or blocked response
        if not response.parts:
             # Check for safety ratings if available and log/report
            if response.prompt_feedback and response.prompt_feedback.block_reason:
                logging.warning(f"Gemini response blocked. Reason: {response.prompt_feedback.block_reason}")
                return f"I apologize, but I encountered an issue generating the response (Content Blocked: {response.prompt_feedback.block_reason}). Please try rephrasing your question.", []
            else:
                 logging.warning("Gemini response was empty or incomplete.")
                 return "I apologize, but I couldn't generate a complete response. Please try again.", []

        answer = response.text

        # Append source citation
        if sources:
            source_citation = f"\n\n*Sources: [{', '.join(sources)}]*"
            answer += source_citation

        return answer

    except Exception as e:
        logging.error(f"Error calling Gemini API: {e}")
        st.error(f"🚨 API Error: Could not get response from Gemini. {e}")
        return "An error occurred while trying to generate the response.", []


# --- Streamlit App ---

# Page configuration
st.set_page_config(
    page_title="Auxilium Corp HRBP",
    page_icon="ℹ️",
    layout="wide"
)

# --- Header ---
# Using columns for layout
col1, col2 = st.columns([1, 4])
with col1:
    # Placeholder for a logo if you have one
     st.image("https://img.icons8.com/fluency/96/chatbot.png", width=96) # Example icon
    # st.image("path/to/your/logo.png", width=100) # Uncomment and replace with your logo path
with col2:
    st.markdown("# Auxilium Corp HRBP")
    st.markdown("##### Your AI Human Resources Business Partner ℹ️")
st.markdown("---")
st.markdown(
    """
    Welcome! I'm here to help you with questions about:
    * 🚀 **Onboarding:** Requirements, first steps, FAQs.
    * 📜 **Company Policies:** Find information on guidelines and procedures.
    * 🏢 **Organizational Culture:** Learn more about our values and work environment.

    Ask me anything!
    """
)
st.markdown("---")


# Load data (cached)
processed_data = load_and_embed_pdfs(PDF_FOLDER_PATH)

# Initialize chat history in session state if it doesn't exist
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle chat input from the user
if prompt := st.chat_input("Ask me about onboarding, policies, or culture..."):
    # Add user message to chat history and display it
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Process the query only if data was loaded successfully
    if processed_data:
        # Find relevant documents
        with st.spinner("Searching our knowledgebase..."):
            best_matches_text, best_matches_sources = find_best_matches(prompt, processed_data)

        # Generate response using Gemini
        if best_matches_text:
             with st.spinner("ℹ️ Thinking..."):
                response_text = get_gemini_response(prompt, best_matches_text, best_matches_sources)
        else:
             response_text = "Based on the knowledgebase, I could not find relevant information to answer that question."
             best_matches_sources = [] # Ensure sources list is empty

        # Add assistant response to chat history and display it
        st.session_state.messages.append({"role": "assistant", "content": response_text})
        with st.chat_message("assistant"):
            st.markdown(response_text)

    else:
        # Handle case where document loading failed earlier
        error_message = "I apologize, but I cannot answer questions right now due to a technical error. Please contact support."
        st.session_state.messages.append({"role": "assistant", "content": error_message})
        with st.chat_message("assistant"):
            st.markdown(error_message)
        st.error("Document loading failed. Please check the 'hr_docs' folder and PDF files.")

# --- Footer ---
st.markdown("---")
st.caption("Auxilium Corp HRBP v1.0 - Powered by Google Gemini")
import os
import streamlit as st
import pandas as pd
from typing import List, Dict, Any

from pdf_processor import PDFProcessor
from embeddings import EmbeddingGenerator
from vector_store import FAISSVectorStore
from chat_model import ChatModel
from utils import (
    check_api_keys,
    initialize_session_state,
    format_chat_history,
    render_chat_message,
)

# Set page configuration
st.set_page_config(
    page_title="Multi-PDF RAG Chat App",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Apply custom CSS
st.markdown("""
<style>
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .stButton button {
        width: 100%;
    }
    .document-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 10px;
        margin-bottom: 10px;
    }
    .source-text {
        color: #0068c9;
        font-weight: bold;
    }
    h1, h2, h3 {
        color: #0068c9;
    }
</style>
""", unsafe_allow_html=True)

# Function to upload and process PDFs
def process_uploaded_pdfs(uploaded_files):
    """Upload and process PDF files."""
    pdf_processor = PDFProcessor()
    embedding_generator = EmbeddingGenerator()
    vector_store = FAISSVectorStore()
    
    # Process each file
    for uploaded_file in uploaded_files:
        # Skip if already processed
        if uploaded_file.name in st.session_state.uploaded_files:
            continue
        
        with st.spinner(f"Processing {uploaded_file.name}..."):
            # Save the uploaded file
            file_path = pdf_processor.save_uploaded_file(uploaded_file)
            
            # Process the PDF
            chunks, filename = pdf_processor.process_pdf(file_path)
            
            # Generate embeddings
            chunks_with_embeddings = embedding_generator.generate_embeddings(chunks)
            
            # Store in vector database
            vector_store.add_documents(chunks_with_embeddings)
            
            # Add to processed files
            st.session_state.uploaded_files.add(uploaded_file.name)
    
    # Get document statistics
    doc_count = vector_store.get_document_count()
    source_docs = vector_store.get_source_documents()
    
    return doc_count, source_docs

# Function to handle chat interaction
def handle_chat(query, llm_provider, llm_model, top_k=5):
    """Process the user query and generate a response."""
    embedding_generator = EmbeddingGenerator()
    vector_store = FAISSVectorStore()
    chat_model = ChatModel(provider=llm_provider, model=llm_model)
    
    # Generate query embedding
    query_embedding = embedding_generator.generate_query_embedding(query)
    
    # Search for relevant documents
    results = vector_store.similarity_search(query_embedding, k=top_k)
    
    # Format chat history for the LLM
    formatted_history = format_chat_history(st.session_state.chat_history[-5:] if len(st.session_state.chat_history) > 5 else st.session_state.chat_history)
    
    # Generate response
    response = chat_model.generate_response(query, results, formatted_history)
    
    # Update chat history
    st.session_state.chat_history.append({"is_user": True, "content": query})
    st.session_state.chat_history.append({"is_user": False, "content": response})
    
    return results

# Main app function
def main():
    """Main application function."""
    # Initialize session state
    initialize_session_state()
    
    # Title and introduction
    st.title("📚 Multi-PDF RAG Chat App")
    
    # Sidebar - Configuration
    with st.sidebar:
        st.header("📝 Configuration")
        
        # API key check
        has_api_keys = check_api_keys()
        
        # Model selection
        llm_provider = st.selectbox("Select LLM Provider", ["OpenAI", "Anthropic"], index=0)
        
        # Model options based on provider
        if llm_provider == "OpenAI":
            llm_model = st.selectbox("Select Model", ["gpt-3.5-turbo", "gpt-4o"], index=0)
        else:
            llm_model = st.selectbox("Select Model", ["claude-3-haiku-20240307", "claude-3-sonnet-20240229", "claude-3-opus-20240229"], index=0)
        
        # Number of results to retrieve
        top_k = st.slider("Number of chunks to retrieve", min_value=1, max_value=10, value=5)
        
        # PDF Upload Section
        st.header("📤 Upload PDFs")
        uploaded_files = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)
        
        if uploaded_files:
            process_button = st.button("Process PDFs")
            if process_button:
                doc_count, source_docs = process_uploaded_pdfs(uploaded_files)
                st.success(f"✅ Processed {len(source_docs)} documents with {doc_count} total chunks!")
        
        # Document Management
        st.header("📋 Document Management")
        vector_store = FAISSVectorStore()
        source_docs = vector_store.get_source_documents()
        
        if source_docs:
            st.write(f"Total documents: {len(source_docs)}")
            st.write(f"Total chunks: {vector_store.get_document_count()}")
            
            # Display managed documents
            for doc in source_docs:
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(f"📄 {doc}")
                with col2:
                    if st.button("Delete", key=f"delete_{doc}"):
                        vector_store.delete_document(doc)
                        st.experimental_rerun()
        else:
            st.info("No documents uploaded yet. Please upload PDFs to begin.")
        
        # Clear chat
        if st.session_state.chat_history:
            if st.button("Clear Chat History"):
                st.session_state.chat_history = []
                st.experimental_rerun()
    
    # Main Content Area - Chat Interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Chat Area
        st.header("💬 Chat with your PDFs")
        
        # Display chat history
        for message in st.session_state.chat_history:
            render_chat_message(message)
        
        # Chat input
        if not source_docs:
            st.info("Please upload and process PDFs before starting the chat.")
        else:
            query = st.chat_input("Ask a question about your documents...")
            if query:
                # Process the query
                results = handle_chat(query, llm_provider.lower(), llm_model, top_k)
                
                # No need to manually display messages here as they're added to session state
                # and will be displayed on the next rerun
    
    with col2:
        # Search Results/Context
        st.header("🔍 Search Context")
        
        if 'chat_history' in st.session_state and len(st.session_state.chat_history) >= 2:
            # Get the last user query
            last_user_query = None
            for message in reversed(st.session_state.chat_history):
                if message["is_user"]:
                    last_user_query = message["content"]
                    break
            
            if last_user_query:
                # Get search results for the query
                embedding_generator = EmbeddingGenerator()
                vector_store = FAISSVectorStore()
                query_embedding = embedding_generator.generate_query_embedding(last_user_query)
                results = vector_store.similarity_search(query_embedding, k=top_k)
                
                if results:
                    st.write(f"Showing top {len(results)} chunks for: '{last_user_query}'")
                    
                    # Display results
                    for i, doc in enumerate(results):
                        with st.expander(f"Chunk {i+1} - {doc.get('metadata', {}).get('source', 'Unknown')}"):
                            st.markdown(f"**Source**: {doc.get('metadata', {}).get('source', 'Unknown')}")
                            st.markdown(f"**Text**: {doc['text']}")
                            st.markdown(f"**Relevance Score**: {doc.get('score', 'N/A'):.4f}")
                else:
                    st.info("No relevant chunks found.")
        else:
            st.info("Ask a question to see relevant document chunks here.")

# Run the app
if __name__ == "__main__":
    main()
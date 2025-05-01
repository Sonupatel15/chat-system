import os
import streamlit as st
from typing import List, Dict, Any, Optional

def check_api_keys():
    """Check if the necessary API keys are set in the environment or Streamlit secrets."""
    # Check for OpenAI API key
    openai_key = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY", "")
    if not openai_key:
        st.warning("⚠️ OpenAI API key not found. Please set it in .streamlit/secrets.toml or as an environment variable.")
    
    # Check for Anthropic API key
    anthropic_key = os.getenv("ANTHROPIC_API_KEY") or st.secrets.get("ANTHROPIC_API_KEY", "")
    if not anthropic_key:
        st.warning("⚠️ Anthropic API key not found. Please set it in .streamlit/secrets.toml or as an environment variable.")
    
    # Return True if at least one key is set
    return bool(openai_key or anthropic_key)

def initialize_session_state():
    """Initialize Streamlit session state variables if they don't exist."""
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    if "uploaded_files" not in st.session_state:
        st.session_state.uploaded_files = set()

def format_chat_history(chat_history: List[Dict]) -> List[Dict]:
    """Format chat history for OpenAI/Anthropic API.
    
    Args:
        chat_history (List[Dict]): List of chat messages
        
    Returns:
        List[Dict]: Formatted chat history
    """
    formatted_history = []
    
    for message in chat_history:
        formatted_history.append({
            "role": "user" if message["is_user"] else "assistant",
            "content": message["content"]
        })
    
    return formatted_history

def render_chat_message(message: Dict):
    """Render a chat message in the Streamlit UI.
    
    Args:
        message (Dict): Chat message with 'is_user' and 'content' keys
    """
    if message["is_user"]:
        st.chat_message("user").write(message["content"])
    else:
        st.chat_message("assistant").write(message["content"])
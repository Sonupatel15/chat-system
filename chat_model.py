import os
from typing import List, Dict, Any, Optional
import openai
from anthropic import Anthropic

class ChatModel:
    def __init__(self, provider: str = "openai", model: Optional[str] = None):
        """Initialize the chat model.
        
        Args:
            provider (str): AI provider ("openai" or "anthropic")
            model (Optional[str]): Model name to use. Defaults to GPT-3.5-turbo for OpenAI and Claude 3 Haiku for Anthropic.
        """
        self.provider = provider.lower()
        
        # Set up OpenAI
        if self.provider == "openai":
            openai.api_key = os.getenv("OPENAI_API_KEY")
            self.model = model or "gpt-3.5-turbo"
        
        # Set up Anthropic
        elif self.provider == "anthropic":
            self.client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
            self.model = model or "claude-3-haiku-20240307"
        
        else:
            raise ValueError(f"Unsupported provider: {provider}. Use 'openai' or 'anthropic'.")
            
    def generate_response(self, query: str, context: List[Dict], chat_history: List[Dict] = None) -> str:
        """Generate a response based on the query, context, and chat history.
        
        Args:
            query (str): User query
            context (List[Dict]): Relevant documents for context
            chat_history (List[Dict]): Previous chat messages
            
        Returns:
            str: Generated response
        """
        # Format context
        context_text = "\n\n".join([f"Document {i+1} (Source: {doc.get('metadata', {}).get('source', 'unknown')}): {doc['text']}" 
                                   for i, doc in enumerate(context)])
        
        # Create prompt
        system_prompt = """You are a helpful AI assistant that answers questions based on the provided context from PDF documents. 
        If the answer cannot be found in the context, acknowledge that and provide general information if possible.
        Always mention the source document when referencing information from the context.
        """
        
        # Process based on provider
        if self.provider == "openai":
            return self._generate_openai_response(system_prompt, query, context_text, chat_history)
        elif self.provider == "anthropic":
            return self._generate_anthropic_response(system_prompt, query, context_text, chat_history)
    
    def _generate_openai_response(self, system_prompt: str, query: str, context_text: str, chat_history: List[Dict] = None) -> str:
        """Generate a response using OpenAI API."""
        messages = [
            {"role": "system", "content": system_prompt}
        ]
        
        # Add chat history if provided
        if chat_history:
            messages.extend(chat_history)
        
        # Add context and query
        messages.append({"role": "user", "content": f"Context:\n{context_text}\n\nQuestion: {query}"})
        
        # Generate response
        response = openai.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.3,
            max_tokens=1000
        )
        
        return response.choices[0].message.content
    
    def _generate_anthropic_response(self, system_prompt: str, query: str, context_text: str, chat_history: List[Dict] = None) -> str:
        """Generate a response using Anthropic API."""
        # Format chat history for Claude
        formatted_history = ""
        if chat_history:
            for msg in chat_history:
                role = "Human" if msg["role"] == "user" else "Assistant"
                formatted_history += f"\n\n{role}: {msg['content']}"
        
        # Create prompt
        prompt = f"{system_prompt}\n\n"
        
        if formatted_history:
            prompt += formatted_history + "\n\n"
            
        prompt += f"Human: Context:\n{context_text}\n\nQuestion: {query}\n\nAssistant:"
        
        # Generate response
        response = self.client.messages.create(
            model=self.model,
            max_tokens=1000,
            temperature=0.3,
            system=system_prompt,
            messages=[
                {"role": "user", "content": f"Context:\n{context_text}\n\nQuestion: {query}"}
            ]
        )
        
        return response.content[0].text
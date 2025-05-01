import os
import PyPDF2
import pdfplumber
import uuid
from typing import List, Dict, Tuple
from langchain.text_splitter import RecursiveCharacterTextSplitter

class PDFProcessor:
    def __init__(self, data_dir: str = "data"):
        """Initialize the PDF processor.
        
        Args:
            data_dir (str): Directory to store uploaded PDFs
        """
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        
        # Text splitter for chunking
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
    
    def save_uploaded_file(self, uploaded_file) -> str:
        """Save an uploaded file to disk and return the path.
        
        Args:
            uploaded_file: Streamlit uploaded file object
            
        Returns:
            str: Path to the saved file
        """
        # Create a unique filename to avoid conflicts
        file_id = str(uuid.uuid4())[:8]
        filename = f"{file_id}_{uploaded_file.name}"
        file_path = os.path.join(self.data_dir, filename)
        
        # Save the file
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
            
        return file_path
    
    def extract_text_pypdf2(self, file_path: str) -> str:
        """Extract text from a PDF using PyPDF2.
        
        Args:
            file_path (str): Path to the PDF file
            
        Returns:
            str: Extracted text
        """
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page_num in range(len(reader.pages)):
                page = reader.pages[page_num]
                text += page.extract_text() + "\n"
        return text
    
    def extract_text_pdfplumber(self, file_path: str) -> str:
        """Extract text from a PDF using pdfplumber.
        
        Args:
            file_path (str): Path to the PDF file
            
        Returns:
            str: Extracted text
        """
        text = ""
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                extracted_text = page.extract_text()
                if extracted_text:
                    text += extracted_text + "\n"
        return text
    
    def extract_text(self, file_path: str) -> str:
        """Extract text from a PDF using multiple methods for robustness.
        
        Args:
            file_path (str): Path to the PDF file
            
        Returns:
            str: Extracted text
        """
        # Try PyPDF2 first
        text = self.extract_text_pypdf2(file_path)
        
        # If we got minimal text, try pdfplumber as backup
        if len(text.strip()) < 100:
            text = self.extract_text_pdfplumber(file_path)
            
        return text
    
    def create_chunks(self, text: str, filename: str) -> List[Dict]:
        """Split text into chunks for embedding.
        
        Args:
            text (str): Text to split
            filename (str): Source filename for metadata
            
        Returns:
            List[Dict]: List of chunks with metadata
        """
        # Split the text into chunks
        chunks = self.text_splitter.split_text(text)
        
        # Add metadata to each chunk
        documents = []
        for i, chunk in enumerate(chunks):
            doc = {
                "id": f"{os.path.basename(filename)}_chunk_{i}",
                "text": chunk,
                "metadata": {
                    "source": os.path.basename(filename),
                    "chunk": i
                }
            }
            documents.append(doc)
            
        return documents
    
    def process_pdf(self, file_path: str) -> Tuple[List[Dict], str]:
        """Process a PDF file: extract text and create chunks.
        
        Args:
            file_path (str): Path to the PDF file
            
        Returns:
            Tuple[List[Dict], str]: (List of chunk documents, filename)
        """
        filename = os.path.basename(file_path)
        text = self.extract_text(file_path)
        chunks = self.create_chunks(text, filename)
        
        return chunks, filename
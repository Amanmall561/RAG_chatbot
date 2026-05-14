import os
import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma

# Local directory for ChromaDB storage
CHROMA_PERSIST_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")

def get_embeddings():
    # Ensure GOOGLE_API_KEY is available in the environment
    return GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

def get_vector_store():
    return Chroma(
        collection_name="documents",
        embedding_function=get_embeddings(),
        persist_directory=CHROMA_PERSIST_DIR
    )

def extract_text_from_bytes(file_bytes: bytes, filename: str) -> str:
    if filename.lower().endswith('.pdf'):
        text = ""
        with fitz.open(stream=file_bytes, filetype="pdf") as doc:
            for page in doc:
                text += page.get_text() + "\n"
        return text
    elif filename.lower().endswith('.txt'):
        return file_bytes.decode('utf-8', errors='ignore')
    else:
        raise ValueError("Unsupported file type. Only PDF and TXT are supported.")

def process_and_store_document(file_bytes: bytes, filename: str) -> int:
    text = extract_text_from_bytes(file_bytes, filename)
    if not text.strip():
        raise ValueError("No text could be extracted from the document.")

    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    
    chunks = text_splitter.create_documents(
        [text], 
        metadatas=[{"source": filename}]
    )
    
    # Store in ChromaDB
    vector_store = get_vector_store()
    vector_store.add_documents(chunks)
    
    return len(chunks)

def retrieve_context(query: str, k: int = 4) -> str:
    vector_store = get_vector_store()
    
    try:
        results = vector_store.similarity_search(query, k=k)
        if not results:
            return "No relevant documents found."
            
        context = ""
        for i, res in enumerate(results):
            source = res.metadata.get("source", "Unknown")
            context += f"--- Document Chunk {i+1} (Source: {source}) ---\n"
            context += res.page_content + "\n\n"
        return context
    except Exception as e:
        return f"Error retrieving documents: {str(e)}"

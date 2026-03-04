import os
from dotenv import load_dotenv
from llama_index.core import SimpleDirectoryReader, Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.fireworks import Fireworks
import logging

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure LlamaIndex global settings
Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
Settings.llm = Fireworks(
    model="accounts/fireworks/models/llama-v3p1-70b-instruct",
    api_key=os.getenv("FIREWORKS_API_KEY"),
    temperature=0.0,  # deterministic for RAG
    max_tokens=512,
)

def load_documents(directory_path: str = "documents"):
    """Load all supported documents from a folder"""
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        logger.warning(f"Created empty '{directory_path}' folder. Add PDFs/TXT files.")
        return []

    logger.info(f"Loading documents from {directory_path}")
    reader = SimpleDirectoryReader(
        input_dir=directory_path,
        required_exts=[".pdf", ".txt", ".docx"],
        recursive=False
    )
    docs = reader.load_data()
    logger.info(f"Loaded {len(docs)} documents")
    return docs

def chunk_documents(docs: list[Document]):
    """Split documents into chunks with metadata preservation"""
    logger.info("Chunking documents...")
    
    splitter = SentenceSplitter(
        chunk_size=512,         # ~300-400 words per chunk
        chunk_overlap=128,      # overlap for context continuity
        paragraph_separator="\n\n|\n|\r\n",
        secondary_chunking_regex=r"[.!?]\s+",
    )
    
    nodes = splitter.get_nodes_from_documents(docs)
    logger.info(f"Created {len(nodes)} chunks")

    # Show sample chunk with metadata
    if nodes:
        sample = nodes[0]
        print("\n=== SAMPLE CHUNK ===")
        print(f"Text (first 200 chars): {sample.text[:200]}...")
        print(f"Metadata: {sample.metadata}")
        print("===================\n")

    return nodes

if __name__ == "__main__":
    docs = load_documents("documents")
    if docs:
        chunks = chunk_documents(docs)
    else:
        print("No documents found. Add some PDFs or TXT files to the 'documents' folder.")
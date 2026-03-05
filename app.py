# app.py – Week 2 Day 2: Chunking + Embeddings + Qdrant Storage
import os
from dotenv import load_dotenv
from llama_index.core import SimpleDirectoryReader, Document, VectorStoreIndex, StorageContext
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.fireworks import Fireworks
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
import logging

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure LlamaIndex
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
Settings.llm = Fireworks(
    model="accounts/fireworks/models/llama-v3p1-70b-instruct",
    api_key=os.getenv("FIREWORKS_API_KEY"),
    temperature=0.0,
    max_tokens=512,
)

# Qdrant client (local on Droplet)
qdrant_client = QdrantClient(path="./qdrant_data")
vector_store = QdrantVectorStore(client=qdrant_client, collection_name="doc_chunks")
storage_context = StorageContext.from_defaults(vector_store=vector_store)

def load_and_index_documents(directory_path: str = "documents"):
    """Load, chunk, embed, and store documents in Qdrant"""
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        logger.warning(f"Created empty '{directory_path}' folder. Add PDFs/TXT files.")
        return None

    logger.info(f"Loading documents from {directory_path}")
    reader = SimpleDirectoryReader(
        input_dir=directory_path,
        required_exts=[".pdf", ".txt", ".docx"],
        recursive=False
    )
    docs = reader.load_data()
    logger.info(f"Loaded {len(docs)} documents")

    # Chunking
    splitter = SentenceSplitter(chunk_size=512, chunk_overlap=128)
    nodes = splitter.get_nodes_from_documents(docs)
    logger.info(f"Created {len(nodes)} chunks")

    # Build index (embeds + stores in Qdrant)
    index = VectorStoreIndex(nodes, storage_context=storage_context, show_progress=True)
    logger.info("Index created and stored in Qdrant")

    # Sample query test
    query_engine = index.as_query_engine()
    sample_response = query_engine.query("What is this document about?")
    print("\nSample query test:")
    print(sample_response)

    return index

if __name__ == "__main__":
    index = load_and_index_documents("documents")
    if index:
        print("Ingestion complete. Chunks embedded and stored in Qdrant.")
    else:
        print("No documents found. Add files to 'documents/' folder.")
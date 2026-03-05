# app.py –  Full RAG Chain with Fireworks + Citations
import os
from dotenv import load_dotenv
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.fireworks import Fireworks
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor
import logging
from llama_index.core import PromptTemplate
from llama_index.core.response_synthesizers import get_response_synthesizer

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure LlamaIndex
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
Settings.llm = Fireworks(
    model="accounts/fireworks/models/llama-v3p3-70b-instruct",
    api_key=os.getenv("FIREWORKS_API_KEY"),
    temperature=0.0,
    max_tokens=1024,
)

# Qdrant setup
qdrant_client = QdrantClient(path="./qdrant_data")
vector_store = QdrantVectorStore(client=qdrant_client, collection_name="doc_chunks")
storage_context = StorageContext.from_defaults(vector_store=vector_store)

def build_or_load_index(directory_path: str = "documents"):
    """Load documents, chunk, embed, and store/index in Qdrant"""
    if os.path.exists("./qdrant_data/collections/doc_chunks"):
        logger.info("Loading existing index from Qdrant")
        index = VectorStoreIndex.from_vector_store(vector_store, storage_context=storage_context)
    else:
        logger.info("Building new index")
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)
            logger.warning(f"Created empty '{directory_path}' folder. Add PDFs/TXT files.")
            return None

        reader = SimpleDirectoryReader(
            input_dir=directory_path,
            required_exts=[".pdf", ".txt", ".docx"],
            recursive=False
        )
        docs = reader.load_data()
        logger.info(f"Loaded {len(docs)} documents")

        splitter = SentenceSplitter(chunk_size=512, chunk_overlap=128)
        nodes = splitter.get_nodes_from_documents(docs)
        logger.info(f"Created {len(nodes)} chunks")

        index = VectorStoreIndex(nodes, storage_context=storage_context, show_progress=True)
        logger.info("Index created and stored in Qdrant")

    return index

def setup_query_engine(index):
    """Create RAG query engine with citation support"""
    retriever = index.as_retriever(similarity_top_k=10)  # retrieve top 10 chunks

    rag_prompt = PromptTemplate(
    "Context from documents:\n{context_str}\n\n"
    "Question: {query_str}\n\n"
    "Answer the question using only the context. "
    "If not enough information, say 'I don't have enough information'. "
    "Include citations with file name and page if available."
    )

    response_synthesizer = get_response_synthesizer(text_qa_template=rag_prompt)

    query_engine = RetrieverQueryEngine(
        retriever=retriever,
        response_synthesizer=response_synthesizer,
        node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.5)]
    )

    return query_engine

if __name__ == "__main__":
    index = build_or_load_index("documents")
    if index:
        query_engine = setup_query_engine(index)

        # Interactive test loop
        print("\nRAG Query Test – type 'exit' to quit")
        while True:
            question = input("Ask a question about the documents: ")
            if question.lower() == "exit":
                break
            if not question.strip():
                continue

            response = query_engine.query(question)
            print("\nAnswer:")
            print(response.response)
            print("\nSources:")
            for node in response.source_nodes:
                print(f"- {node.node.metadata.get('file_name', 'Unknown')} (page {node.node.metadata.get('page_label', 'N/A')})")
                print(f"  Score: {node.score:.3f}")
                print(f"  Text preview: {node.node.text[:200]}...\n")
    else:
        print("No documents found. Add files to 'documents/' folder.")
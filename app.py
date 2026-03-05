# app.py – Gradio Web UI + RAG with Upload-Only Ingestion
import os
import gradio as gr
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
from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.core import PromptTemplate
import logging
import tempfile

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

# Qdrant setup (persistent on disk)
qdrant_client = QdrantClient(path="./qdrant_data")
vector_store = QdrantVectorStore(client=qdrant_client, collection_name="doc_chunks")
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# Global index (lazy loaded or created on first upload)
index = None

def initialize_index():
    """Create or load Qdrant index (only called once)"""
    global index
    if index is None:
        logger.info("Initializing Qdrant index")
        index = VectorStoreIndex.from_vector_store(
            vector_store, storage_context=storage_context
        )
    return index

def ingest_and_query(file, question):
    """Upload file → ingest → query"""
    global index
    answer = ""
    sources = ""

    try:
        # 1. Ingest new file if uploaded
        if file:
            # Gradio file is a path → use directly
            tmp_path = file.name

            docs = SimpleDirectoryReader(input_files=[tmp_path]).load_data()
            splitter = SentenceSplitter(chunk_size=512, chunk_overlap=128)
            nodes = splitter.get_nodes_from_documents(docs)

            # Initialize index if first upload
            initialize_index()

            # Insert new nodes
            index.insert_nodes(nodes)

            answer += f"**File ingested:** {os.path.basename(tmp_path)} ({len(nodes)} chunks added)\n\n"

        # 2. Answer question if asked
        if question.strip():
            # Make sure index is initialized
            initialize_index()

            query_engine = index.as_query_engine(
                similarity_top_k=10,
                node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.4)],
                response_synthesizer=get_response_synthesizer(
                    text_qa_template=PromptTemplate(
                        "Context from documents:\n{context_str}\n\n"
                        "Question: {query_str}\n\n"
                        "Answer using only the context. "
                        "If not enough information, say 'I don't have enough information'. "
                        "Include citations with file name and page if available."
                    )
                )
            )

            response = query_engine.query(question)
            answer += f"**Answer:** {response.response}\n\n"
            sources = "**Sources & Citations:**\n"
            for node in response.source_nodes:
                sources += f"- **{node.node.metadata.get('file_name', 'Unknown')}** (page {node.node.metadata.get('page_label', 'N/A')})\n"
                sources += f"  Score: {node.score:.3f}\n"
                sources += f"  Preview: {node.node.text[:200]}...\n\n"

        return answer, sources

    except Exception as e:
        return f"Error: {str(e)}", ""

# Gradio interface
with gr.Blocks(title="DocMind RAG") as demo:
    gr.Markdown("# DocMind RAG – Private Document Q&A")
    gr.Markdown("Upload documents → ask questions → get cited answers (all in cloud)")

    with gr.Row():
        file_input = gr.File(label="Upload PDF, TXT or DOCX", file_types=[".pdf", ".txt", ".docx"])
        question_input = gr.Textbox(label="Ask a question", lines=3, placeholder="e.g. What is Majd's experience?")

    submit_btn = gr.Button("Submit")

    output_answer = gr.Textbox(label="Answer", lines=10, interactive=False)
    output_sources = gr.Textbox(label="Sources / Citations", lines=10, interactive=False)

    submit_btn.click(
        ingest_and_query,
        inputs=[file_input, question_input],
        outputs=[output_answer, output_sources]
    )

    gr.Markdown("Note: First upload builds the index. Subsequent uploads add to it. No local folder used.")

demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
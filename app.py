# app.py – Gradio UI + Auth + Logging + Clear Qdrant Button

import os
import sys
import gradio as gr
from dotenv import load_dotenv
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.fireworks import Fireworks
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.core import PromptTemplate

# Logging
from loguru import logger

# Load environment variables
load_dotenv()
APP_API_KEY = os.getenv("APP_API_KEY")

# -----------------------------
# Logging configuration
# -----------------------------
logger.remove()

logger.add(
    "logs/app_{time}.log",
    rotation="500 MB",
    retention="10 days",
    level="INFO",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"
)

logger.add(sys.stdout, level="DEBUG")

# -----------------------------
# LlamaIndex configuration
# -----------------------------
Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5"
)

Settings.llm = Fireworks(
    model="accounts/fireworks/models/llama-v3p3-70b-instruct",
    api_key=os.getenv("FIREWORKS_API_KEY"),
    temperature=0.0,
    max_tokens=1024,
)

# -----------------------------
# Qdrant configuration
# -----------------------------
qdrant_path = "./qdrant_data"

qdrant_client = QdrantClient(path=qdrant_path)

vector_store = QdrantVectorStore(
    client=qdrant_client,
    collection_name="doc_chunks"
)

storage_context = StorageContext.from_defaults(
    vector_store=vector_store
)

# Global index
index = None


# -----------------------------
# Initialize index
# -----------------------------
def initialize_index():
    global index

    if index is None:
        logger.info("Initializing Qdrant index")

        index = VectorStoreIndex.from_vector_store(
            vector_store,
            storage_context=storage_context
        )

    return index


# -----------------------------
# Clear Qdrant safely
# -----------------------------
def clear_qdrant():

    global index, qdrant_client, vector_store, storage_context

    try:

        logger.info("Clearing Qdrant collection")

        try:
            qdrant_client.delete_collection("doc_chunks")
        except Exception:
            pass

        # recreate vector store
        vector_store = QdrantVectorStore(
            client=qdrant_client,
            collection_name="doc_chunks"
        )

        storage_context = StorageContext.from_defaults(
            vector_store=vector_store
        )

        index = None

        logger.info("Qdrant reset completed")

        return "✅ Vector database cleared. Upload a new document."

    except Exception as e:
        logger.error(f"Error clearing Qdrant: {e}")
        return f"Error clearing Qdrant: {str(e)}"


# -----------------------------
# Ingest and Query
# -----------------------------
def ingest_and_query(file, question, api_key):

    if api_key != APP_API_KEY:
        return "Invalid API key.", ""

    initialize_index()

    answer = ""
    sources = ""

    try:

        # -----------------
        # Ingest document
        # -----------------
        if file:

            tmp_path = file.name

            docs = SimpleDirectoryReader(
                input_files=[tmp_path]
            ).load_data()

            splitter = SentenceSplitter(
                chunk_size=512,
                chunk_overlap=128
            )

            nodes = splitter.get_nodes_from_documents(docs)

            index.insert_nodes(nodes)

            answer += f"File ingested: {os.path.basename(tmp_path)} ({len(nodes)} chunks added)\n\n"

        # -----------------
        # Query
        # -----------------
        if question.strip():

            query_engine = index.as_query_engine(

                similarity_top_k=10,

                node_postprocessors=[
                    SimilarityPostprocessor(similarity_cutoff=0.4)
                ],

                response_synthesizer=get_response_synthesizer(

                    text_qa_template=PromptTemplate(

                        "Context from documents:\n{context_str}\n\n"
                        "Question: {query_str}\n\n"
                        "Answer using only the context. "
                        "If not enough information say "
                        "'I don't have enough information'. "
                        "Include citations."

                    )
                )
            )

            response = query_engine.query(question)

            answer += f"Answer:\n{response.response}\n\n"

            sources = "Sources:\n"

            for node in response.source_nodes:

                sources += f"- {node.node.metadata.get('file_name','Unknown')}"
                sources += f" (page {node.node.metadata.get('page_label','N/A')})\n"
                sources += f"Score: {node.score:.3f}\n"
                sources += f"Preview: {node.node.text[:200]}...\n\n"

        return answer, sources

    except Exception as e:

        logger.error(f"Error in ingest/query: {e}")

        return f"Error: {str(e)}", ""


# -----------------------------
# Gradio UI
# -----------------------------
with gr.Blocks(title="DocMind RAG") as demo:

    gr.Markdown("# DocMind RAG – Private Document Q&A")

    gr.Markdown(
        "Upload documents → ask questions → get cited answers"
    )

    api_key_input = gr.Textbox(
        label="API Key",
        type="password"
    )

    file_input = gr.File(
        label="Upload PDF, TXT or DOCX",
        file_types=[".pdf", ".txt", ".docx"]
    )

    question_input = gr.Textbox(
        label="Ask a question",
        lines=3
    )

    submit_btn = gr.Button("Submit")

    clear_btn = gr.Button("Clear Vector Database")

    output_answer = gr.Textbox(
        label="Answer",
        lines=10,
        interactive=False
    )

    output_sources = gr.Textbox(
        label="Sources / Citations",
        lines=10,
        interactive=False
    )

    submit_btn.click(
        ingest_and_query,
        inputs=[file_input, question_input, api_key_input],
        outputs=[output_answer, output_sources]
    )

    clear_btn.click(
        clear_qdrant,
        outputs=output_answer
    )

    gr.Markdown(
        "API key required. Clear the database to start fresh."
    )


demo.launch(
    server_name="0.0.0.0",
    server_port=7860,
    share=False
)
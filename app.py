# app.py – Week 2 Day 5: Gradio UI + Auth + Logging + Clear Qdrant Button
# Enhancements: Chat history, multiple file upload, loading indicator, index stats

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
import logging
import tempfile

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

# Chat history (list of {"role": "user/assistant", "content": "..."})
chat_history = []

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
    global index, qdrant_client, vector_store, storage_context, chat_history

    try:
        logger.info("Clearing Qdrant collection")
        try:
            qdrant_client.delete_collection("doc_chunks")
        except Exception:
            pass

        # Recreate vector store
        vector_store = QdrantVectorStore(
            client=qdrant_client,
            collection_name="doc_chunks"
        )

        storage_context = StorageContext.from_defaults(
            vector_store=vector_store
        )

        index = None
        chat_history = []  # clear chat history too

        logger.info("Qdrant reset completed")
        return "✅ Vector database cleared. Upload a new document."

    except Exception as e:
        logger.error(f"Error clearing Qdrant: {e}")
        return f"Error clearing Qdrant: {str(e)}"


# -----------------------------
# Get current index stats
# -----------------------------
def get_index_stats():
    try:
        count = qdrant_client.get_collection("doc_chunks").points_count
        return f"Current knowledge base: {count} chunks"
    except:
        return "Current knowledge base: 0 chunks"


# -----------------------------
# Ingest and Query (with chat history & multiple files)
# -----------------------------
def ingest_and_query(files, question, api_key, progress=gr.Progress()):

    if api_key != APP_API_KEY:
        return "Invalid API key.", "", get_index_stats(), chat_history

    initialize_index()

    answer = ""
    sources = ""

    try:
        progress(0.1, desc="Processing...")

        # -----------------
        # Ingest multiple files
        # -----------------
        if files:
            progress(0.3, desc="Ingesting files...")
            all_nodes = []
            for file in files:
                tmp_path = file.name
                docs = SimpleDirectoryReader(input_files=[tmp_path]).load_data()
                splitter = SentenceSplitter(chunk_size=512, chunk_overlap=128)
                nodes = splitter.get_nodes_from_documents(docs)
                all_nodes.extend(nodes)

            if all_nodes:
                index.insert_nodes(all_nodes)
                answer += f"**Ingested {len(files)} file(s)** ({len(all_nodes)} chunks added)\n\n"

        progress(0.6, desc="Generating answer...")

        # -----------------
        # Query with chat history
        # -----------------
        if question.strip():
            # Build context from recent history
            history_str = ""
            for msg in chat_history[-6:]:  # last 6 messages for context
                role = msg["role"]
                content = msg["content"]
                history_str += f"{role.capitalize()}: {content}\n"

            full_query = f"{history_str}User: {question}"

            query_engine = index.as_query_engine(
                similarity_top_k=10,
                node_postprocessors=[
                    SimilarityPostprocessor(similarity_cutoff=0.4)
                ],
                response_synthesizer=get_response_synthesizer(
                    text_qa_template=PromptTemplate(
                        "Conversation history:\n{history_str}\n\n"
                        "Context from documents:\n{context_str}\n\n"
                        "Question: {query_str}\n\n"
                        "Answer using only the context and history. "
                        "If not enough information say "
                        "'I don't have enough information'. "
                        "Include citations."
                    ).partial_format(history_str=history_str)
                )
            )

            response = query_engine.query(question)
            answer += f"**Answer:** {response.response}\n\n"

            sources = "**Sources:**\n"
            for node in response.source_nodes:
                sources += f"- {node.node.metadata.get('file_name','Unknown')}"
                sources += f" (page {node.node.metadata.get('page_label','N/A')})\n"
                sources += f"Score: {node.score:.3f}\n"
                sources += f"Preview: {node.node.text[:200]}...\n\n"

            # Append to chat history
            chat_history.append({"role": "user", "content": question})
            chat_history.append({"role": "assistant", "content": response.response})

        progress(1.0, desc="Done!")

        return answer, sources, get_index_stats(), chat_history

    except Exception as e:
        logger.error(f"Error in ingest/query: {e}")
        return f"Error: {str(e)}", "", get_index_stats(), chat_history


# -----------------------------
# Gradio UI
# -----------------------------
with gr.Blocks(title="DocMind RAG") as demo:

    gr.Markdown("# DocMind RAG – Private Document Q&A")
    gr.Markdown("Upload documents → ask questions → get cited answers")

    api_key_input = gr.Textbox(
        label="API Key",
        type="password"
    )

    file_input = gr.File(
        label="Upload one or multiple PDF/TXT/DOCX",
        file_types=[".pdf", ".txt", ".docx"],
        file_count="multiple"
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

    index_stats = gr.Textbox(
        label="Current Knowledge Base",
        interactive=False,
        value=get_index_stats()
    )

    chatbot = gr.Chatbot(label="Conversation History")

    submit_btn.click(
        ingest_and_query,
        inputs=[file_input, question_input, api_key_input],
        outputs=[output_answer, output_sources, index_stats, chatbot]
    )

    clear_btn.click(
        clear_qdrant,
        outputs=output_answer
    ).then(
        lambda: get_index_stats(),
        outputs=index_stats
    )

    gr.Markdown(
        "API key required. Clear the database to start fresh."
    )


demo.launch(
    server_name="0.0.0.0",
    server_port=7860,
    share=False
)
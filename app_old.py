"""
app.py – Gradio UI + Auth + Logging + Clear Qdrant Button

Refactored to improve readability, internal structure, and performance.
"""

import os
import sys
import tempfile
from typing import List, Dict, Any, Tuple, Optional

import gradio as gr
from dotenv import load_dotenv
from loguru import logger

from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    StorageContext,
    Settings,
    PromptTemplate,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.fireworks import Fireworks
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.response_synthesizers import get_response_synthesizer


# Type alias for chat messages to improve type hinting
ChatHistory = List[Dict[str, str]]


class AppConfig:
    """
    Data class storing configuration variables and initializing global settings.
    # [Structure] Separates environment loading and constant configuration from business logic.
    """

    def __init__(self) -> None:
        load_dotenv()
        self.api_key: str = os.getenv("APP_API_KEY", "")
        self.fireworks_api_key: str = os.getenv("FIREWORKS_API_KEY", "")
        self.qdrant_path: str = "./qdrant_data"
        self.collection_name: str = "doc_chunks"

        # [Performance] Configure LlamaIndex global settings only once on startup instead of repeatedly.
        Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
        Settings.llm = Fireworks(
            model="accounts/fireworks/models/llama-v3p3-70b-instruct",
            api_key=self.fireworks_api_key,
            temperature=0.0,
            max_tokens=1024,
        )


class DocumentProcessor:
    """
    Handles internal logic for parsing and chunking files.
    """

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 128) -> None:
        # [Performance] Instantiate SentenceSplitter once to avoid redundant initialization per document.
        self.splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    def process_files(self, file_paths: List[str]) -> List[Any]:
        """
        Reads multiple documents and splits them into nodes.
        
        Args:
            file_paths: A list of absolute file paths to ingest.
            
        Returns:
            A list of node objects parsed from the given documents.
        """
        # [Performance] Group files into a single SimpleDirectoryReader call for optimized batch I/O.
        docs = SimpleDirectoryReader(input_files=file_paths).load_data()
        return self.splitter.get_nodes_from_documents(docs)


class RAGManager:
    """
    Encapsulates Qdrant logic, LlamaIndex querying, and chat state.
    # [Structure] Removes global database clients and chat history by containing them in this class.
    """

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.qdrant_client = QdrantClient(path=self.config.qdrant_path)
        self.vector_store = QdrantVectorStore(
            client=self.qdrant_client, collection_name=self.config.collection_name
        )
        self.storage_context = StorageContext.from_defaults(vector_store=self.vector_store)

        self.index: Optional[VectorStoreIndex] = None
        self.chat_history: ChatHistory = []
        self.doc_processor = DocumentProcessor()

        # [Performance] Predefine the prompt string once to avoid recreating it in every query.
        self.base_qa_template_str = (
            "Conversation history:\n{history_str}\n\n"
            "Context from documents:\n{context_str}\n\n"
            "Question: {query_str}\n\n"
            "Answer using only the context and history. "
            "If not enough information say 'I don't have enough information'. "
            "Include citations."
        )

    def _initialize_index(self) -> None:
        """Lazily connects LlamaIndex to the underlying Qdrant vector store."""
        if self.index is None:
            logger.info("Initializing Qdrant index")
            self.index = VectorStoreIndex.from_vector_store(
                self.vector_store, storage_context=self.storage_context
            )

    def clear_database(self) -> Tuple[str, ChatHistory]:
        """
        Deletes the Qdrant collection and resets index state completely.
        
        Returns:
            A tuple containing a status string and the cleared chat history.
        """
        logger.info("Clearing Qdrant collection")
        try:
            self.qdrant_client.delete_collection(self.config.collection_name)
        except Exception as e:
            # [Readability] Log the exception string locally instead of doing a bare 'pass'.
            logger.debug(f"Collection custom delete skipped or failed: {e}")

        # Re-initialize the vector store after deletion to prevent broken references
        self.vector_store = QdrantVectorStore(
            client=self.qdrant_client, collection_name=self.config.collection_name
        )
        self.storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
        self.index = None
        self.chat_history.clear()

        logger.info("Qdrant reset completed")
        return "✅ Vector database cleared. Upload a new document.", self.chat_history

    def get_stats(self) -> str:
        """Counts the current document chunks indexed in Qdrant."""
        try:
            count = self.qdrant_client.get_collection(self.config.collection_name).points_count
            return f"Current knowledge base: {count} chunks"
        except Exception:
            # If the collection doesn't exist yet, it returns 0
            return "Current knowledge base: 0 chunks"

    def _format_history_string(self) -> str:
        """
        # [Readability] Extracted history parsing from the main pipeline to a dedicated helper.
        """
        history_str = ""
        for msg in self.chat_history[-6:]:
            role = msg.get("role", "system")
            content = msg.get("content", "")
            history_str += f"{role.capitalize()}: {content}\n"
        return history_str

    def ingest_and_query(
        self,
        files: Optional[List[Any]],
        question: str,
        api_key: str,
        progress: gr.Progress = gr.Progress()
    ) -> Tuple[str, str, str, ChatHistory]:
        """
        The main pipeline to ingest files (if provided) and answer a generic text query over them.
        
        Args:
            files: Gradio file objects representing the uploaded attachments.
            question: The user query string.
            api_key: Text input API key for pseudo-authentication.
            progress: Gradio Progress tracker.
            
        Returns:
            A tuple of (short answer, details block, stats block, new chat history).
        """
        if api_key != self.config.api_key:
            return "**Invalid API key.**", "", self.get_stats(), self.chat_history

        self._initialize_index()
        answer = ""
        details = ""

        try:
            progress(0.1, desc="Processing...")

            # Phase 1: Ingestion
            if files:
                progress(0.3, desc="Ingesting files...")
                file_paths = [file.name for file in files]
                new_nodes = self.doc_processor.process_files(file_paths)

                if new_nodes:
                    # 'index' shouldn't be None per earlier initialization
                    if self.index:
                        self.index.insert_nodes(new_nodes)
                    details += f"Ingested {len(files)} file(s) ({len(new_nodes)} chunks added)\n\n"

            progress(0.6, desc="Generating answer...")

            # Phase 2: Generating Answer
            if question.strip() and self.index is not None:
                history_str = self._format_history_string()
                
                # Format the base template securely and dynamically
                qa_template = PromptTemplate(self.base_qa_template_str).partial_format(
                    history_str=history_str
                )

                query_engine = self.index.as_query_engine(
                    similarity_top_k=10,
                    node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.4)],
                    response_synthesizer=get_response_synthesizer(text_qa_template=qa_template)
                )

                # Query execution
                response = query_engine.query(question)
                clean_answer = response.response.strip()

                answer = f"**{clean_answer}**"
                details += "**Sources:**\n"

                for node in response.source_nodes:
                    metadata = node.node.metadata
                    file_name = metadata.get("file_name", "Unknown")
                    page_label = metadata.get("page_label", "N/A")

                    details += f"- {file_name} (page {page_label})\n"
                    details += f"  Score: {node.score:.3f}\n"
                    details += f"  Preview: {node.node.text[:200]}...\n\n"

                self.chat_history.append({"role": "user", "content": question})
                self.chat_history.append({"role": "assistant", "content": clean_answer})

            progress(1.0, desc="Done!")
            return answer, details, self.get_stats(), self.chat_history

        except Exception as e:
            logger.error(f"Error in ingest/query: {e}")
            return f"**Error:** {str(e)}", "", self.get_stats(), self.chat_history


class RAGAppUI:
    """
    Constructs and launches the Gradio user interface.
    # [Structure] Separates the UI configuration from the application core.
    """

    def __init__(self, manager: RAGManager) -> None:
        self.manager = manager

    def build_and_launch(self) -> None:
        """Defines the frontend view and attaches the respective component signals."""
        with gr.Blocks(title="DocMind RAG") as demo:
            gr.Markdown("# DocMind RAG – Private Document Q&A")
            gr.Markdown("Upload documents → ask questions → get cited answers")

            with gr.Row():
                api_key_input = gr.Textbox(label="API Key", type="password", scale=1)

            with gr.Row():
                file_input = gr.File(
                    label="Upload one or multiple PDF/TXT/DOCX",
                    file_types=[".pdf", ".txt", ".docx"],
                    file_count="multiple",
                    scale=1
                )

            gr.Markdown("### Conversation")

            chatbot = gr.Chatbot(label="Chat History", height=400)

            with gr.Row():
                question_input = gr.Textbox(
                    label="Ask a question",
                    lines=2,
                    placeholder="Type your question here...",
                    scale=4
                )
                submit_btn = gr.Button("Send", scale=1)

            clear_btn = gr.Button("Clear Vector Database")

            output_details = gr.Textbox(
                label="Details / Sources / Ingestion Info",
                lines=8,
                interactive=False
            )

            index_stats = gr.Textbox(
                label="Current Knowledge Base",
                interactive=False,
                value=self.manager.get_stats()
            )

            # Signal wiring
            submit_btn.click(
                self.manager.ingest_and_query,
                inputs=[file_input, question_input, api_key_input],
                outputs=[output_details, output_details, index_stats, chatbot]
            )

            clear_btn.click(
                self.manager.clear_database,
                outputs=output_details
            ).then(
                self.manager.get_stats,
                outputs=index_stats
            ).then(
                lambda: [],  # Clear chat history locally in UI
                outputs=chatbot
            )

            gr.Markdown("API key required. Clear the database to start fresh.")

        # Main loop runner
        demo.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False
        )


def main() -> None:
    """Bootstraps the application by configuring services and starting the UI loop."""
    # Centralized logging setup
    logger.remove()
    logger.add("logs/app_{time}.log", rotation="500 MB", retention="10 days", level="INFO")
    logger.add(sys.stdout, level="DEBUG")

    config = AppConfig()
    manager = RAGManager(config)
    app_ui = RAGAppUI(manager)
    
    app_ui.build_and_launch()


if __name__ == "__main__":
    main()
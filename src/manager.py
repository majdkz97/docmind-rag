from typing import Dict, List, Tuple, Optional, Any
import gradio as gr
from loguru import logger

from llama_index.core import VectorStoreIndex, StorageContext, PromptTemplate
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

from src.config import AppConfig
from src.document_processor import DocumentProcessor

# Type alias for chat messages to improve type hinting
ChatHistory = List[Dict[str, str]]


class RAGManager:
    """
    Encapsulates Qdrant logic, LlamaIndex querying, and chat state.
    Removes global database clients and chat history by containing them in this class.
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

        # Predefine the prompt string once to avoid recreating it in every query.
        self.base_qa_template_str = (
            "Conversation history:\n{history_str}\n\n"
            "Context from documents:\n{context_str}\n\n"
            "Question: {query_str}\n\n"
            "Answer the question using only the provided context and conversation history. "
            "If the context does not contain enough information to answer the question, just say 'I don't have enough information'.\n"
            "Organize your answer clearly (using paragraphs or bullet points if appropriate).\n"
            "When citing sources, you MUST use only the base file name (e.g. 'document.pdf') and the page number. "
            "NEVER include full directory paths (like /tmp/gradio/...) in your citations or output. "
            "Format citations like this: [filename.pdf, Page X]."
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
            # Log the exception string locally instead of doing a bare 'pass'.
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
        Extracted history parsing from the main pipeline to a dedicated helper.
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
                current_file_paths = [file.name for file in files]
                
                # Check if we actually have new files to ingest
                # If they are exactly the same as the last ones, skip clearing and re-ingesting!
                if not hasattr(self, "last_ingested_files") or current_file_paths != self.last_ingested_files:
                    progress(0.2, desc="Clearing previous database...")
                    # Automatically format the database so new inputs don't mix with old ones
                    # This ALSO clears the chat history to start a fresh session!
                    self.clear_database()
                    # Must re-initialize index since clear_database() drops it 
                    self._initialize_index()
                    
                    progress(0.3, desc="Ingesting files...")
                    new_nodes = self.doc_processor.process_files(current_file_paths)

                    if new_nodes:
                        # 'index' shouldn't be None per earlier initialization
                        if self.index:
                            self.index.insert_nodes(new_nodes)
                        details += f"Ingested {len(files)} file(s) ({len(new_nodes)} chunks added)\n\n"
                    
                    self.last_ingested_files = current_file_paths

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

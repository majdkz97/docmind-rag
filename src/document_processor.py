from typing import List, Any
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter


class DocumentProcessor:
    """
    Handles internal logic for parsing and chunking files.
    """

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 128) -> None:
        # Instantiate SentenceSplitter once to avoid redundant initialization per document.
        self.splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    def process_files(self, file_paths: List[str]) -> List[Any]:
        """
        Reads multiple documents and splits them into nodes.
        
        Args:
            file_paths: A list of absolute file paths to ingest.
            
        Returns:
            A list of node objects parsed from the given documents.
        """
        # Group files into a single SimpleDirectoryReader call for optimized batch I/O.
        docs = SimpleDirectoryReader(input_files=file_paths).load_data()
        
        # Scrub long absolute server paths from metadata so the LLM doesn't include them in citations
        for doc in docs:
            if "file_path" in doc.metadata:
                # Replace the full /tmp/gradio/... path with just the base file name
                doc.metadata["file_path"] = doc.metadata.get("file_name", "Unknown Document")
                
        return self.splitter.get_nodes_from_documents(docs)

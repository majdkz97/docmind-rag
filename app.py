# app.py – Week 2 Day 4: Gradio Web UI + Full RAG with Upload & Citations
import os
import gradio as gr
from dotenv import load_dotenv
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext, PromptTemplate
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.fireworks import Fireworks
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.response_synthesizers import get_response_synthesizer
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

def get_or_build_index():
    """Load existing Qdrant index or build new one from documents folder"""
    if os.path.exists("./qdrant_data/collections/doc_chunks"):
        logger.info("Loading existing Qdrant index")
        return VectorStoreIndex.from_vector_store(vector_store, storage_context=storage_context)
    
    logger.info("Building new index from documents folder")
    docs = SimpleDirectoryReader(input_dir="documents", required_exts=[".pdf", ".txt", ".docx"]).load_data()
    if not docs:
        raise ValueError("No documents found in 'documents' folder.")

    splitter = SentenceSplitter(chunk_size=512, chunk_overlap=128)
    nodes = splitter.get_nodes_from_documents(docs)
    index = VectorStoreIndex(nodes, storage_context=storage_context, show_progress=True)
    logger.info("New index built and stored in Qdrant")
    return index

# Global index (loaded once at startup)
index = get_or_build_index()

# Custom RAG prompt
rag_prompt = PromptTemplate(
    "Context from documents:\n{context_str}\n\n"
    "Question: {query_str}\n\n"
    "Answer the question using only the context above. "
    "If not enough information, say 'I don't have enough information'. "
    "Include citations with file name and page number if available."
)

# Build query engine with custom prompt
response_synthesizer = get_response_synthesizer(text_qa_template=rag_prompt)

query_engine = RetrieverQueryEngine(
    retriever=index.as_retriever(similarity_top_k=10),
    response_synthesizer=response_synthesizer,
    node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.5)]
)

def ingest_and_query(file, question):
    """Upload file → ingest → query"""
    answer = ""
    sources = ""

    try:
        # 1. Ingest new file if uploaded
        if file:
            # Gradio file is a NamedString or dict-like object
            # Save it to temporary disk path
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.name)[1]) as tmp:
                # Write the file content (file.content is bytes)
                tmp.write(file.content)
                tmp_path = tmp.name

            # Now load with LlamaIndex
            docs = SimpleDirectoryReader(input_files=[tmp_path]).load_data()
            splitter = SentenceSplitter(chunk_size=512, chunk_overlap=128)
            nodes = splitter.get_nodes_from_documents(docs)

            # Insert new nodes into existing index
            index.insert_nodes(nodes)
            os.unlink(tmp_path)  # clean up temp file

            answer += f"New file ingested: {file.name} ({len(nodes)} chunks added)\n\n"

        # 2. Answer question if asked
        if question.strip():
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
    gr.Markdown("Upload documents → ask questions → get answers with citations")

    with gr.Row():
        with gr.Column(scale=1):
            file_input = gr.File(label="Upload PDF, TXT or DOCX", file_types=[".pdf", ".txt", ".docx"])
        with gr.Column(scale=3):
            question_input = gr.Textbox(label="Ask a question about all documents", lines=3, placeholder="e.g. What is the main topic?")
            submit_btn = gr.Button("Submit")

    output_answer = gr.Textbox(label="Answer", lines=12, interactive=False)
    output_sources = gr.Textbox(label="Sources / Citations", lines=10, interactive=False)

    submit_btn.click(
        ingest_and_query,
        inputs=[file_input, question_input],
        outputs=[output_answer, output_sources]
    )

    gr.Markdown("Note: First upload may take time to embed. Answers are grounded in uploaded documents only.")

# Launch web server
demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
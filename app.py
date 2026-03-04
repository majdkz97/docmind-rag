# app.py – Hugging Face Space for document chunking (Week 2 Day 1)
import gradio as gr
import os
from dotenv import load_dotenv
from llama_index.core import SimpleDirectoryReader, Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.fireworks import Fireworks

# Load environment variables
load_dotenv()

# Configure LlamaIndex
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
Settings.llm = Fireworks(
    model="accounts/fireworks/models/llama-v3p1-70b-instruct",
    api_key=os.getenv("FIREWORKS_API_KEY"),
    temperature=0.0,
    max_tokens=512,
)

def chunk_document(file):
    if not file:
        return "Please upload a file.", ""

    try:
        # Save uploaded file temporarily
        with open(file.name, "wb") as f:
            f.write(file.read())

        # Load document
        reader = SimpleDirectoryReader(input_files=[file.name])
        docs = reader.load_data()

        # Chunk
        splitter = SentenceSplitter(chunk_size=512, chunk_overlap=128)
        nodes = splitter.get_nodes_from_documents(docs)

        # Format output
        output = f"Loaded 1 document\nCreated {len(nodes)} chunks\n\nSample chunk:\n"
        if nodes:
            sample = nodes[0]
            output += f"Text preview: {sample.text[:300]}...\n\n"
            output += "Metadata:\n" + "\n".join(f"  {k}: {v}" for k, v in sample.metadata.items())

        os.remove(file.name)  # clean up
        return output, "Success"

    except Exception as e:
        return f"Error: {str(e)}", ""

# Gradio interface
demo = gr.Interface(
    fn=chunk_document,
    inputs=gr.File(label="Upload PDF or TXT"),
    outputs=[
        gr.Textbox(label="Chunking Result", lines=10),
        gr.Textbox(label="Status")
    ],
    title="DocMind RAG – Day 1: Document Chunking",
    description="Upload a document → see how it's chunked for RAG processing. Powered by Fireworks.ai + LlamaIndex."
)

demo.launch(server_name="0.0.0.0", server_port=7860)
# DocMind RAG

**Private Document Question-Answering Chatbot** powered by **Retrieval-Augmented Generation (RAG)**

Upload your PDFs, TXT, or DOCX files → ask natural language questions → get accurate answers with **exact citations** (file name + page number).

Built as part of my **AI Engineer learning journey** (March 2026).

## 🚀 Live Demo

SOON

- **API Key required** — ask me for the key if you're testing  
- Upload one or multiple documents → ask follow-up questions → clear knowledge base anytime

## Features

- **Fully private RAG pipeline** — documents never leave your server  
- **Multiple file upload** — ingest many documents at once  
- **Chat-style interface** — persistent conversation history  
- **Accurate citations** — every answer shows source file, page, score & text preview  
- **Clear Vector Database** button — reset everything with one click  
- **API Key protection** — prevents unauthorized access  
- **Structured logging** — all activity saved to rotating log files  
- **100% cloud-based** — runs on DigitalOcean Droplet (no local execution needed)

## Tech Stack

- **RAG Framework**: LlamaIndex  
- **Embedding model**: BAAI/bge-small-en-v1.5 (Hugging Face)  
- **LLM**: Llama 3.1 70B via Fireworks.ai  
- **Vector Database**: Qdrant (local persistent mode)  
- **Web UI**: Gradio  
- **Deployment**: DigitalOcean Droplet (Ubuntu 24.04)  
- **Logging**: Loguru (file rotation + console)  
- **Environment**: Python 3.12 + venv

## Quick Start (on your own Droplet)

1. **Clone the repo**
   ```bash
   git clone https://github.com/majdkz97/docmind-rag.git
   cd docmind-rag
   ```

2. **Set up virtual environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Create `.env` file**
   ```bash
   nano .env
   ```
   Add your keys:
   ```
   FIREWORKS_API_KEY=fw_XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
   APP_API_KEY=your-super-secret-key-here
   ```

4. **Run the app**
   ```bash
   python app.py
   ```

5. **Open in browser**
   http://YOUR_DROPLET_IP:7860

   - Enter API key  
   - Upload documents  
   - Start asking questions!

## Project Structure

```
docmind-rag/
├── app.py               # Main Gradio app + full RAG pipeline
├── requirements.txt     # All dependencies
├── .env                 # Secrets (never commit!)
├── logs/                # Rotating log files (auto-created)
├── qdrant_data/         # Persistent Qdrant vector store (auto-created)
└── README.md
```

## Planned Future Improvements

- Add reranking for even better relevance  
- Support for web URLs (fetch PDFs online)  
- Persistent chat sessions across browser refreshes  
- Export/import knowledge base  
- Docker deployment + Nginx reverse proxy  
- Multi-user support with different collections

## Acknowledgments

- LlamaIndex team  
- Fireworks.ai (fast inference & generous free tier)  
- Hugging Face (amazing open models)  
- Qdrant (excellent local vector DB)  
- Gradio (easiest way to build interactive UIs)

Built with ❤️ during my AI Engineer transition – March 2026

Feedback, suggestions & pull requests welcome!
```
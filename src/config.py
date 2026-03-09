import os

from dotenv import load_dotenv
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.fireworks import Fireworks

class AppConfig:
    """
    Data class storing configuration variables and initializing global settings.
    Separates environment loading and constant configuration from business logic.
    """

    def __init__(self) -> None:
        load_dotenv()
        self.api_key: str = os.getenv("APP_API_KEY", "")
        self.fireworks_api_key: str = os.getenv("FIREWORKS_API_KEY", "")
        self.qdrant_path: str = "./qdrant_data"
        self.collection_name: str = "doc_chunks"

        # Configure LlamaIndex global settings only once on startup instead of repeatedly.
        Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
        Settings.llm = Fireworks(
            model="accounts/fireworks/models/llama-v3p3-70b-instruct",
            api_key=self.fireworks_api_key,
            temperature=0.0,
            max_tokens=1024,
        )

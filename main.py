import sys
from loguru import logger

from src.config import AppConfig
from src.manager import RAGManager
from src.ui import RAGAppUI

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

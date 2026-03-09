import gradio as gr
from src.manager import RAGManager


class RAGAppUI:
    """
    Constructs and launches the Gradio user interface.
    Separates the UI configuration from the application core.
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

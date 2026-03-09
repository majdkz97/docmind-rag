import gradio as gr
from src.manager import RAGManager


class RAGAppUI:
    """
    Constructs and launches the Gradio user interface.
    Separates the UI configuration from the application core.
    """

    def __init__(self, manager: RAGManager) -> None:
        self.manager = manager
        self.custom_css = """
        /* Base Colors (Light Mode) */
        :root {
            --bg-color: #F9FAFB;
            --surface-color: #FFFFFF;
            --border-color: #E5E7EB;
            --text-primary: #111827;
            --bot-bg: #FFFFFF;
            --upload-bg: #F9FAFB;
            --upload-hover: #EEF2FF;
        }

        /* Dark Mode Colors (Gradio naturally adds .dark to body) */
        body.dark {
            --bg-color: #0b0f19;
            --surface-color: #111827;
            --border-color: #374151;
            --text-primary: #F9FAFB;
            --bot-bg: #1f2937;
            --upload-bg: #111827;
            --upload-hover: #1f2937;
        }

        /* Overall Background and Font */
        body, .gradio-container {
            background-color: var(--bg-color) !important;
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
            color: var(--text-primary) !important;
        }

        /* Sidebar styling */
        .sidebar-column {
            background-color: var(--surface-color) !important;
            border-right: 1px solid var(--border-color);
            padding: 24px !important;
            border-radius: 12px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.2);
            height: 100%;
        }

        /* Chatbot Area */
        .gradio-chatbot {
            background-color: transparent !important;
            border: none !important;
        }

        /* User Message Bubble */
        .user-row .message {
            background: linear-gradient(135deg, #4F46E5 0%, #6366F1 100%) !important;
            color: white !important;
            border-radius: 16px 16px 4px 16px !important;
            box-shadow: 0 4px 6px -1px rgba(79, 70, 229, 0.2) !important;
            font-size: 15px;
            line-height: 1.6;
        }

        /* Assistant Message Bubble */
        .bot-row .message {
            background-color: var(--bot-bg) !important;
            border: 1px solid var(--border-color) !important;
            color: var(--text-primary) !important;
            border-radius: 16px 16px 16px 4px !important;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1) !important;
            font-size: 15px;
            line-height: 1.6;
        }

        /* File Upload Box (Drag and Drop zone) */
        .upload-container {
            border: 2px dashed var(--border-color) !important;
            border-radius: 12px !important;
            background-color: var(--upload-bg) !important;
            transition: border-color 0.2s ease, background-color 0.2s ease !important;
        }
        .upload-container:hover {
            border-color: #4F46E5 !important;
            background-color: var(--upload-hover) !important;
        }

        /* Smooth Slide-in Animation for chat messages */
        @keyframes slideUpFade {
            0% { opacity: 0; transform: translateY(10px); }
            100% { opacity: 1; transform: translateY(0); }
        }
        .message {
            animation: slideUpFade 0.3s ease-out forwards;
        }
        """

    def build_and_launch(self) -> None:
        """Defines the frontend view and attaches the respective component signals."""
        theme = gr.themes.Default(primary_hue="indigo", neutral_hue="slate")
        
        with gr.Blocks(title="DocMind RAG") as demo:
            with gr.Row():
                # Sidebar
                with gr.Column(scale=1, elem_classes="sidebar-column"):
                    gr.Markdown("## 🧠 DocMind RAG\n*Private Document Q&A*")
                    
                    gr.Markdown("### ⚙️ Settings")
                    api_key_input = gr.Textbox(label="API Key", type="password", placeholder="Enter your proxy API key...")
                    
                    gr.Markdown("### 📄 Knowledge Base")
                    file_input = gr.File(
                        label="Upload one or multiple documents",
                        file_types=[".pdf", ".txt", ".docx"],
                        file_count="multiple",
                        elem_classes="upload-container"
                    )
                    
                    index_stats = gr.Textbox(
                        label="Current Knowledge Base",
                        interactive=False,
                        value=self.manager.get_stats()
                    )
                    
                    clear_btn = gr.Button("Clear Vector Database", variant="secondary")
                    
                    with gr.Accordion("Ingestion Details & Sources", open=False):
                        output_details = gr.Textbox(
                            show_label=False,
                            lines=10,
                            interactive=False
                        )
                
                # Main Chat Area
                with gr.Column(scale=3):
                    chatbot = gr.Chatbot(
                        label="Chat History", 
                        height=600
                    )
                    
                    with gr.Row(equal_height=True):
                        question_input = gr.Textbox(
                            show_label=False,
                            lines=2,
                            placeholder="Type your question here (Press Enter to send)...",
                            scale=4
                        )
                        submit_btn = gr.Button("Send", variant="primary", scale=1)

            # Signal wiring
            submit_btn.click(
                self.manager.ingest_and_query,
                inputs=[file_input, question_input, api_key_input],
                outputs=[output_details, output_details, index_stats, chatbot]
            )

            # Allow submitting with Enter key 
            question_input.submit(
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

        # Main loop runner
        demo.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False,
            theme=theme,
            css=self.custom_css
        )

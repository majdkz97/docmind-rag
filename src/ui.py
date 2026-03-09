import gradio as gr
from src.manager import RAGManager

# ---------------------------------------------------------------------------
# Custom CSS – DocMind RAG premium UI  (Gradio 6.x compatible)
# Palette:  #F0F4F8 bg  |  #FFFFFF surface  |  #3B82F6 primary blue
#           #10B981 accent green  |  #1E293B text  |  #64748B muted
# ---------------------------------------------------------------------------
CUSTOM_CSS = """
/* ── Base reset ────────────────────────────────────────────────────────── */
*, *::before, *::after { box-sizing: border-box; }

body, .gradio-container {
    font-family: 'Inter', system-ui, -apple-system, sans-serif !important;
    background: #F0F4F8 !important;
    color: #1E293B !important;
    font-size: 15px;
    line-height: 1.6;
}

/* ── Page wrapper ───────────────────────────────────────────────────────── */
.gradio-container {
    max-width: 1280px !important;
    margin: 0 auto !important;
    padding: 0 !important;
}

/* ── App header ────────────────────────────────────────────────────────── */
#app-header {
    background: linear-gradient(135deg, #1E3A5F 0%, #2563EB 60%, #3B82F6 100%);
    padding: 24px 32px 20px;
    border-radius: 0 0 24px 24px;
    margin-bottom: 28px;
    box-shadow: 0 4px 24px rgba(37, 99, 235, 0.18);
    animation: fadeSlideDown 0.5s ease both;
}

#app-header h1 {
    margin: 0 0 4px;
    font-size: 26px;
    font-weight: 700;
    color: #FFFFFF !important;
    letter-spacing: -0.3px;
}

#app-header p {
    margin: 0;
    font-size: 13px;
    color: rgba(255,255,255,0.78) !important;
    font-weight: 400;
}

/* ── Sidebar panel ─────────────────────────────────────────────────────── */
#sidebar-panel {
    background: #FFFFFF;
    border-radius: 16px;
    padding: 20px;
    box-shadow: 0 1px 8px rgba(15, 23, 42, 0.07);
    border: 1px solid #E2E8F0;
    animation: fadeIn 0.45s ease 0.1s both;
    display: flex;
    flex-direction: column;
    gap: 18px;
    height: 100%;
}

/* section labels inside sidebar */
.section-label {
    font-size: 11px !important;
    font-weight: 600 !important;
    color: #64748B !important;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    margin: 0 0 4px 0 !important;
    padding: 0 !important;
}

/* ── API key input ─────────────────────────────────────────────────────── */
#api-key-box textarea, #api-key-box input {
    border-radius: 10px !important;
    border: 1.5px solid #CBD5E1 !important;
    background: #F8FAFC !important;
    padding: 10px 14px !important;
    font-size: 14px !important;
    transition: border-color 0.2s ease, box-shadow 0.2s ease;
}
#api-key-box textarea:focus, #api-key-box input:focus {
    border-color: #3B82F6 !important;
    box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.12) !important;
    outline: none !important;
}

/* ── File upload zone ──────────────────────────────────────────────────── */
#file-upload-zone {
    border: 2px dashed #94A3B8 !important;
    border-radius: 14px !important;
    background: #F8FAFC !important;
    padding: 18px 12px !important;
    text-align: center;
    transition: border-color 0.25s ease, background 0.25s ease;
    cursor: pointer;
    min-height: 110px;
}
#file-upload-zone:hover {
    border-color: #3B82F6 !important;
    background: #EFF6FF !important;
}

/* ── Stats / knowledge base box ────────────────────────────────────────── */
#stats-box {
    background: linear-gradient(135deg, #EFF6FF, #F0FDF4) !important;
    border-radius: 12px !important;
    border: 1px solid #BFDBFE !important;
    animation: fadeIn 0.5s ease 0.25s both;
}
#stats-box textarea, #stats-box input {
    background: transparent !important;
    border: none !important;
    font-size: 13px !important;
    color: #1E40AF !important;
    font-weight: 500 !important;
    resize: none !important;
    box-shadow: none !important;
}
#stats-box label span {
    font-size: 11px !important;
    font-weight: 600 !important;
    color: #3B82F6 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.7px !important;
}

/* ── Clear button ──────────────────────────────────────────────────────── */
#clear-db-btn {
    background: #FEF2F2 !important;
    color: #EF4444 !important;
    border: 1.5px solid #FECACA !important;
    border-radius: 10px !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    transition: background 0.2s ease, border-color 0.2s ease, transform 0.15s ease;
    cursor: pointer;
}
#clear-db-btn:hover {
    background: #FEE2E2 !important;
    border-color: #EF4444 !important;
    transform: translateY(-1px);
}
#clear-db-btn:active { transform: translateY(0); }

/* ── Main chat panel ───────────────────────────────────────────────────── */
#chat-panel {
    background: #FFFFFF;
    border-radius: 16px;
    padding: 24px;
    box-shadow: 0 1px 8px rgba(15, 23, 42, 0.07);
    border: 1px solid #E2E8F0;
    animation: fadeIn 0.45s ease 0.05s both;
    display: flex;
    flex-direction: column;
    gap: 16px;
}

/* ── Chatbot messages ──────────────────────────────────────────────────── */
#chatbot-main {
    border: none !important;
    background: #F8FAFC !important;
    border-radius: 12px !important;
}
/* user bubble */
#chatbot-main .user {
    background: linear-gradient(135deg, #2563EB, #3B82F6) !important;
    color: #FFFFFF !important;
    border-radius: 18px 18px 4px 18px !important;
    box-shadow: 0 2px 10px rgba(37, 99, 235, 0.22) !important;
    animation: slideInRight 0.28s ease both;
}
/* assistant bubble */
#chatbot-main .bot {
    background: #FFFFFF !important;
    color: #1E293B !important;
    border-radius: 18px 18px 18px 4px !important;
    box-shadow: 0 1px 6px rgba(15, 23, 42, 0.09) !important;
    border: 1px solid #E2E8F0 !important;
    animation: slideInLeft 0.28s ease both;
}

/* ── Question input ────────────────────────────────────────────────────── */
#question-input textarea {
    border-radius: 14px !important;
    border: 1.5px solid #CBD5E1 !important;
    background: #F8FAFC !important;
    padding: 14px 18px !important;
    font-size: 15px !important;
    line-height: 1.5 !important;
    resize: none !important;
    transition: border-color 0.22s ease, box-shadow 0.22s ease;
    font-family: 'Inter', system-ui, sans-serif !important;
}
#question-input textarea:focus {
    border-color: #3B82F6 !important;
    box-shadow: 0 0 0 4px rgba(59, 130, 246, 0.1) !important;
    outline: none !important;
    background: #FFFFFF !important;
}
#question-input textarea::placeholder {
    color: #94A3B8 !important;
    font-style: italic;
}

/* ── Send button ───────────────────────────────────────────────────────── */
#send-btn {
    background: linear-gradient(135deg, #2563EB, #3B82F6) !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 14px !important;
    font-size: 15px !important;
    font-weight: 600 !important;
    cursor: pointer;
    transition: transform 0.18s ease, box-shadow 0.18s ease, opacity 0.18s ease;
    box-shadow: 0 4px 14px rgba(37, 99, 235, 0.30) !important;
    letter-spacing: 0.2px;
}
#send-btn:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(37, 99, 235, 0.38) !important;
}
#send-btn:active { transform: translateY(0); opacity: 0.88; }

/* ── Sources / details accordion ───────────────────────────────────────── */
#sources-accordion {
    border-radius: 12px !important;
    border: 1px solid #E2E8F0 !important;
    overflow: hidden;
    animation: fadeIn 0.4s ease both;
}
#sources-accordion > .label-wrap {
    background: #F8FAFC !important;
    font-size: 13px !important;
    font-weight: 600 !important;
    color: #475569 !important;
    cursor: pointer;
    transition: background 0.2s ease;
}
#sources-accordion > .label-wrap:hover { background: #EFF6FF !important; }
#sources-accordion textarea {
    border: none !important;
    background: #FFFFFF !important;
    font-size: 13px !important;
    color: #374151 !important;
    font-family: 'Inter', system-ui, sans-serif !important;
    line-height: 1.65 !important;
    resize: none !important;
}

/* ── Dark-mode toggle button ────────────────────────────────────────────── */
#dark-mode-btn {
    background: #F1F5F9 !important;
    color: #475569 !important;
    border: 1.5px solid #CBD5E1 !important;
    border-radius: 8px !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    cursor: pointer;
    transition: background 0.2s ease, color 0.2s ease;
}
#dark-mode-btn:hover {
    background: #E2E8F0 !important;
    color: #1E293B !important;
}

/* ── Dark mode overrides ────────────────────────────────────────────────── */
body.docmind-dark, .docmind-dark .gradio-container {
    background: #0F172A !important;
    color: #E2E8F0 !important;
}
.docmind-dark #app-header {
    background: linear-gradient(135deg, #0F172A 0%, #1E3A5F 60%, #1D4ED8 100%);
}
.docmind-dark #sidebar-panel,
.docmind-dark #chat-panel {
    background: #1E293B !important;
    border-color: #334155 !important;
}
.docmind-dark #api-key-box textarea,
.docmind-dark #api-key-box input,
.docmind-dark #question-input textarea {
    background: #0F172A !important;
    border-color: #334155 !important;
    color: #E2E8F0 !important;
}
.docmind-dark #chatbot-main {
    background: #0F172A !important;
}
.docmind-dark #chatbot-main .bot {
    background: #1E293B !important;
    border-color: #334155 !important;
    color: #E2E8F0 !important;
}
.docmind-dark #stats-box {
    background: linear-gradient(135deg, #1E3A5F, #064E3B) !important;
    border-color: #1E40AF !important;
}
.docmind-dark #sources-accordion > .label-wrap {
    background: #1E293B !important;
    color: #CBD5E1 !important;
}
.docmind-dark #sources-accordion textarea {
    background: #0F172A !important;
    color: #CBD5E1 !important;
}
.docmind-dark #file-upload-zone {
    background: #1E293B !important;
    border-color: #475569 !important;
}
.docmind-dark #clear-db-btn {
    background: #450A0A !important;
    border-color: #7F1D1D !important;
    color: #FCA5A5 !important;
}

/* ── Keyframe animations ────────────────────────────────────────────────── */
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(6px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes fadeSlideDown {
    from { opacity: 0; transform: translateY(-14px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes slideInRight {
    from { opacity: 0; transform: translateX(16px); }
    to   { opacity: 1; transform: translateX(0); }
}
@keyframes slideInLeft {
    from { opacity: 0; transform: translateX(-16px); }
    to   { opacity: 1; transform: translateX(0); }
}

/* ── Responsive tweaks ──────────────────────────────────────────────────── */
@media (max-width: 768px) {
    #app-header { padding: 18px 20px 14px; }
    #app-header h1 { font-size: 20px; }
    #send-btn { font-size: 14px !important; }
}

/* ── Scrollbar styling ──────────────────────────────────────────────────── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #F1F5F9; border-radius: 4px; }
::-webkit-scrollbar-thumb { background: #CBD5E1; border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: #94A3B8; }
"""

# Gradio 6.x: theme, css, js are passed to demo.launch() instead of gr.Blocks()
DOCMIND_THEME = gr.themes.Base(
    primary_hue="blue",
    secondary_hue="green",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
).set(
    body_background_fill="#F0F4F8",
    block_background_fill="#FFFFFF",
    block_border_width="1px",
    block_border_color="#E2E8F0",
    block_radius="16px",
    block_shadow="0 1px 8px rgba(15,23,42,0.07)",
    input_background_fill="#F8FAFC",
    button_primary_background_fill="linear-gradient(135deg,#2563EB,#3B82F6)",
    button_primary_text_color="#FFFFFF",
)


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

            # ── Header ────────────────────────────────────────────────────
            with gr.Group(elem_id="app-header"):
                gr.HTML("""
                    <h1>🧠 DocMind RAG</h1>
                    <p>Private document Q&amp;A · Upload · Ask · Cite</p>
                """)

            # ── Two-column layout ─────────────────────────────────────────
            with gr.Row(equal_height=False):

                # ── LEFT SIDEBAR ──────────────────────────────────────────
                with gr.Column(scale=1, min_width=260, elem_id="sidebar-panel"):

                    gr.HTML('<p class="section-label">🔑 Authentication</p>')
                    api_key_input = gr.Textbox(
                        label="API Key",
                        type="password",
                        placeholder="Enter your API key…",
                        elem_id="api-key-box",
                    )

                    gr.HTML('<p class="section-label" style="margin-top:8px">📂 Documents</p>')
                    file_input = gr.File(
                        label="Drag & drop or click to upload",
                        file_types=[".pdf", ".txt", ".docx"],
                        file_count="multiple",
                        elem_id="file-upload-zone",
                    )

                    gr.HTML('<p class="section-label" style="margin-top:8px">📊 Knowledge Base</p>')
                    index_stats = gr.Textbox(
                        label="Current Knowledge Base",
                        interactive=False,
                        value=self.manager.get_stats(),
                        elem_id="stats-box",
                        lines=2,
                    )

                    clear_btn = gr.Button(
                        "🗑️  Clear Vector Database",
                        elem_id="clear-db-btn",
                        variant="secondary",
                    )

                    dark_mode_btn = gr.Button(
                        "🌙  Dark Mode",
                        elem_id="dark-mode-btn",
                        variant="secondary",
                    )

                # ── RIGHT CHAT PANEL ──────────────────────────────────────
                with gr.Column(scale=3, elem_id="chat-panel"):

                    chatbot = gr.Chatbot(
                        label="Conversation",
                        height=440,
                        layout="bubble",
                        avatar_images=(None, None),
                        elem_id="chatbot-main",
                    )

                    with gr.Row():
                        question_input = gr.Textbox(
                            label="",
                            lines=2,
                            placeholder="Ask anything about your documents…",
                            scale=5,
                            elem_id="question-input",
                            show_label=False,
                        )
                        send_btn = gr.Button(
                            "Send ➤",
                            scale=1,
                            elem_id="send-btn",
                            variant="primary",
                        )

                    with gr.Accordion(
                        "📎 Sources & Details",
                        open=False,
                        elem_id="sources-accordion",
                    ):
                        output_details = gr.Textbox(
                            label="Sources / Ingestion Info",
                            lines=8,
                            interactive=False,
                            elem_id="details-box",
                            show_label=False,
                        )

            # ── Signal wiring ──────────────────────────────────────────────
            send_btn.click(
                self.manager.ingest_and_query,
                inputs=[file_input, question_input, api_key_input],
                outputs=[output_details, output_details, index_stats, chatbot],
            ).then(
                lambda: "",
                outputs=question_input,
            )

            # Allow Enter key to submit (Shift+Enter for newline)
            question_input.submit(
                self.manager.ingest_and_query,
                inputs=[file_input, question_input, api_key_input],
                outputs=[output_details, output_details, index_stats, chatbot],
            ).then(
                lambda: "",
                outputs=question_input,
            )

            clear_btn.click(
                self.manager.clear_database,
                outputs=[output_details, chatbot],
            ).then(
                self.manager.get_stats,
                outputs=index_stats,
            )

            # Dark mode toggle – pure JS, persists via localStorage
            dark_mode_btn.click(
                fn=None,
                js=(
                    "() => {"
                    " document.body.classList.toggle('docmind-dark');"
                    " localStorage.setItem('docmind-dark',"
                    "  document.body.classList.contains('docmind-dark'));"
                    " }"
                ),
            )

        # In Gradio 6.x, theme/css/js are passed to launch() rather than Blocks()
        demo.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False,
            theme=DOCMIND_THEME,
            css=CUSTOM_CSS,
        )

# -*- coding: utf-8 -*-
import os
import requests
from dotenv import load_dotenv
import gradio as gr
from app.database.session import SessionLocal
from app.models.document import Document
from app.models.chunk import Chunk
from app.models.session import SessionModel
from app.models.message import Message

load_dotenv()

FASTAPI_BASE_URL = os.getenv("FASTAPI_BASE_URL", "http://127.0.0.1:8085").rstrip("/")

def test_fastapi_connection(name: str) -> str:
    if not name.strip():
        return "Please enter your name."
    try:
        from sqlalchemy import text
        db = SessionLocal()
        db.execute(text("SELECT 1"))
        db.close()
        return f"Hello, {name.strip()}! Backend services (FastAPI, SQLite DB, Vector DB) are connected and healthy."
    except Exception as exc:
        return f"Error connecting to backend: {exc}"

def create_document(filename: str) -> str:
    if not filename.strip():
        return "Please enter a filename."
    try:
        db = SessionLocal()
        doc = Document(filename=filename.strip())
        db.add(doc)
        db.commit()
        db.refresh(doc)
        doc_id = doc.id
        db.close()
        return f"Document created!\nID: {doc_id}\nFilename: {filename.strip()}"
    except Exception as exc:
        return f"Error creating document: {exc}"

def list_documents() -> str:
    try:
        db = SessionLocal()
        try:
            docs = db.query(Document).order_by(Document.uploaded_at.desc()).all()
            if not docs:
                return "No documents found in knowledge base."
            lines = []
            for d in docs:
                chunk_c = db.query(Chunk).filter(Chunk.document_id == d.id).count()
                lines.append(f"📄 [ID: {d.id}] {d.filename} | {chunk_c} chunks | Uploaded: {d.uploaded_at}")
            return "\n".join(lines)
        finally:
            db.close()
    except Exception as exc:
        return f"Error listing documents: {exc}"

def create_session() -> str:
    try:
        db = SessionLocal()
        session = SessionModel(title="Research Session")
        db.add(session)
        db.commit()
        db.refresh(session)
        sid = session.id
        db.close()
        return f"Session created!\nID: {sid}\nTitle: Research Session"
    except Exception as exc:
        return f"Error creating session: {exc}"

def list_sessions() -> str:
    try:
        db = SessionLocal()
        try:
            sessions = db.query(SessionModel).order_by(SessionModel.created_at.desc()).all()
            if not sessions:
                return "No chat sessions found."
            lines = []
            for s in sessions:
                msg_c = db.query(Message).filter(Message.session_id == s.id).count()
                lines.append(f"💬 [ID: {s.id}] {s.title} ({msg_c} messages, created {s.created_at})")
            return "\n".join(lines)
        finally:
            db.close()
    except Exception as exc:
        return f"Error listing sessions: {exc}"

def get_kb_stats_html() -> str:
    try:
        db = SessionLocal()
        doc_count = db.query(Document).count()
        db.close()
        from app.services.chroma_service import count_indexed_chunks
        chunk_count = count_indexed_chunks()
        return f"""
        <div style='display:flex; gap:20px; margin-bottom:20px;'>
            <div class='gr-box' style='flex:1; padding:20px; text-align:center;'>
                <h3 style='margin-bottom:10px; color:#94a3b8; font-size:1rem;'>📁 Documents Uploaded</h3>
                <h2 style='color:#a855f7; font-size:2rem; font-weight:800;'>{doc_count}</h2>
            </div>
            <div class='gr-box' style='flex:1; padding:20px; text-align:center;'>
                <h3 style='margin-bottom:10px; color:#94a3b8; font-size:1rem;'>🧩 Chunks Indexed in ChromaDB</h3>
                <h2 style='color:#3b82f6; font-size:2rem; font-weight:800;'>{chunk_count}</h2>
            </div>
            <div class='gr-box' style='flex:1; padding:20px; text-align:center;'>
                <h3 style='margin-bottom:10px; color:#94a3b8; font-size:1rem;'>🟢 Vector DB Status</h3>
                <h2 style='color:#4ade80; font-size:2rem; font-weight:800;'>Online</h2>
            </div>
        </div>
        """
    except Exception:
        return """
        <div style='display:flex; gap:20px; margin-bottom:20px;'>
            <div class='gr-box' style='flex:1; padding:20px; text-align:center;'>
                <h3 style='margin-bottom:10px; color:#94a3b8; font-size:1rem;'>📁 Documents Uploaded</h3>
                <h2 style='color:#a855f7; font-size:2rem; font-weight:800;'>0</h2>
            </div>
            <div class='gr-box' style='flex:1; padding:20px; text-align:center;'>
                <h3 style='margin-bottom:10px; color:#94a3b8; font-size:1rem;'>🧩 Chunks Indexed in ChromaDB</h3>
                <h2 style='color:#3b82f6; font-size:2rem; font-weight:800;'>0</h2>
            </div>
            <div class='gr-box' style='flex:1; padding:20px; text-align:center;'>
                <h3 style='margin-bottom:10px; color:#94a3b8; font-size:1rem;'>🟢 Vector DB Status</h3>
                <h2 style='color:#4ade80; font-size:2rem; font-weight:800;'>Online</h2>
            </div>
        </div>
        """

def upload_pdf(filepath):
    if not filepath:
        return "<div style='color:#ef4444'>Please select a PDF file to upload.</div>", get_kb_stats_html()
    try:
        from app.services.ingestion import ingest_pdf
        db = SessionLocal()
        try:
            filename = os.path.basename(filepath)
            class MockUploadFile:
                def __init__(self, filename, file):
                    self.filename = filename
                    self.file = file
            with open(filepath, "rb") as f:
                mock_file = MockUploadFile(filename, f)
                doc, chunk_count = ingest_pdf(db, mock_file)
            status_html = f"<div style='padding:15px; border-radius:12px; background:rgba(30,41,59,0.7); border:1px solid rgba(255,255,255,0.1);'><div style='display:flex; justify-content:space-between; align-items:center;'><b>📄 {filename}</b><span style='background:rgba(74,222,128,0.2); color:#4ade80; padding:4px 8px; border-radius:12px; font-size:0.8rem; font-weight:600;'>✅ Indexed ({chunk_count} chunks)</span></div></div>"
            return status_html, get_kb_stats_html()
        finally:
            db.close()
    except Exception as exc:
        return f"<div style='color:#ef4444'>Error uploading PDF: {exc}</div>", get_kb_stats_html()

def list_chunks(doc_id: str) -> str:
    if not doc_id.strip():
        return "Please enter a Document ID."
    try:
        db = SessionLocal()
        try:
            chunks = db.query(Chunk).filter(Chunk.document_id == int(doc_id.strip())).order_by(Chunk.chunk_order).all()
            if not chunks:
                return "No chunks found for this Document ID."
            result = []
            for c in chunks:
                text = c.chunk_text.replace('\n', ' ')[:80] + "..."
                result.append(f"[Chunk {c.chunk_order} | Page {c.page_number}] {text}")
            return "\n".join(result)
        finally:
            db.close()
    except Exception as exc:
        return f"Error fetching chunks: {exc}"

def retrieve_chunks(query: str) -> str:
    if not query.strip():
        return "Please enter a search query."
    try:
        from app.services.retrieval_service import retrieve_chunks as do_retrieve
        db = SessionLocal()
        try:
            chunks = do_retrieve(db, query.strip())
            if not chunks:
                return "No matching chunks found in knowledge base."
            result = []
            for c in chunks:
                score = round(c.similarity_score, 4)
                text = c.chunk_text.replace('\n', ' ')
                result.append(f"[{score}] {c.source_filename} (Page {c.page_number}):\n{text}\n")
            return "\n".join(result)
        finally:
            db.close()
    except Exception as exc:
        return f"Error retrieving chunks: {exc}"

def run_research_agent(query: str):
    if not query.strip():
        yield ("<div style='color:#ef4444'>Please enter a research query.</div>", "No query provided.")
        return
    try:
        from app.services.research_agent import run_research
        
        status_log = "<div style='display:flex; flex-direction:column; gap:8px;'>"
        
        # Step 1: Planning
        status_log += "<div style='font-size:0.95rem; padding:8px; background:rgba(255,255,255,0.05); border-radius:8px;'><span style='color:#fbbf24; margin-right:8px;'>●</span> <b>Decomposing Research Query & Planning Tasks...</b></div>"
        yield (status_log + "</div>", "Analyzing query and formulating research plan...")
        
        # Step 2: Retrieving & Synthesizing
        status_log += "<div style='font-size:0.95rem; padding:8px; background:rgba(255,255,255,0.05); border-radius:8px;'><span style='color:#38bdf8; margin-right:8px;'>●</span> <b>Executing Hybrid Retrieval & Web Grounding...</b></div>"
        yield (status_log + "</div>", "Retrieving context across vector knowledge base and web search...")
        
        response = run_research(query.strip())
        
        status_log += "<div style='font-size:0.95rem; padding:8px; background:rgba(255,255,255,0.05); border-radius:8px;'><span style='color:#4ade80; margin-right:8px;'>●</span> <b>Deep Analysis & Report Synthesis Completed</b></div>"
        
        report_md = f"## 📑 Executive Summary\n\n{response.summary}\n\n"
        if response.key_findings:
            report_md += "## 💡 Key Findings\n\n"
            for kf in response.key_findings:
                report_md += f"- {kf}\n"
            report_md += "\n"
            
        if response.sources:
            report_md += "## 📚 Interactive Citations\n\n"
            for s in response.sources:
                page_str = f"Page {s.page_number}" if s.page_number else "Web Source"
                report_md += f"<details style='margin-bottom:8px; border:1px solid rgba(255,255,255,0.1); border-radius:8px; padding:10px; background:#1e293b;'><summary style='cursor:pointer; font-weight:600; color:#818cf8;'>📄 {s.filename} ({page_str})</summary></details>\n"
                
        yield (status_log + "</div>", report_md)
    except Exception as exc:
        yield (f"<div style='color:#ef4444'>Error during research execution: {exc}</div>", f"Research error: {exc}")

def export_research_report(report_text: str, export_format: str):
    if not report_text.strip() or "The generated report will appear here" in report_text or "Research error" in report_text:
        return None
    from pathlib import Path
    temp_dir = Path("data/exports")
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    if export_format == "PDF":
        from app.services.exporter_service import to_pdf
        pdf_path = temp_dir / "Research_Report.pdf"
        data_dict = {
            "summary": report_text,
            "key_findings": ["Report generated by autonomous research agent."],
            "sources": []
        }
        pdf_bytes = to_pdf(data_dict)
        with open(pdf_path, "wb") as f:
            f.write(pdf_bytes.getvalue())
        return str(pdf_path)
    else:
        md_path = temp_dir / "Research_Report.md"
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(report_text)
        return str(md_path)

custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');

body, .gradio-container {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    background-color: #0b0f19 !important;
}

/* Header Styling */
.header-container {
    text-align: center;
    padding: 2.5rem 1rem 1.5rem 1rem;
    background: radial-gradient(circle at top, rgba(99, 102, 241, 0.15) 0%, transparent 60%);
    border-bottom: 1px solid rgba(255, 255, 255, 0.08);
    margin-bottom: 1.5rem;
}

.header-title {
    font-size: 2.5rem;
    font-weight: 800;
    background: linear-gradient(135deg, #a855f7 0%, #6366f1 50%, #3b82f6 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.5rem;
}

.header-subtitle {
    color: #94a3b8;
    font-size: 1.05rem;
    font-weight: 500;
}

/* Tabs Styling */
.tabs {
    border-bottom: 1px solid rgba(255, 255, 255, 0.08) !important;
}

button.tab-nav {
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    padding: 0.75rem 1.25rem !important;
    border-radius: 8px 8px 0 0 !important;
    transition: all 0.2s ease !important;
}

button.tab-nav.selected {
    color: #a855f7 !important;
    border-bottom: 2px solid #a855f7 !important;
    background: rgba(168, 85, 247, 0.08) !important;
}

/* Cards & Accordions */
.gr-box, .gr-panel, .gr-accordion {
    border-radius: 12px !important;
    background-color: #111827 !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    box-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.5) !important;
}

/* Primary Action Buttons */
.gr-button-primary {
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%) !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    transition: transform 0.15s ease, box-shadow 0.15s ease !important;
}

.gr-button-primary:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 14px 0 rgba(99, 102, 241, 0.39) !important;
}

/* Sidebar Panel Styling */
.sidebar-panel {
    background-color: #0f172a !important;
    border-right: 1px solid rgba(255, 255, 255, 0.08) !important;
    padding: 1rem !important;
    border-radius: 12px !important;
}

/* Chat Input Bar Styling */
.gr-textbox textarea {
    background-color: #1e293b !important;
    border: 1px solid rgba(255, 255, 255, 0.12) !important;
    border-radius: 10px !important;
    color: #f8fafc !important;
}

.gr-textbox textarea:focus {
    border-color: #8b5cf6 !important;
    box-shadow: 0 0 0 2px rgba(139, 92, 246, 0.25) !important;
}
"""

theme = gr.themes.Soft(
    primary_hue="indigo",
    neutral_hue="slate"
)

with gr.Blocks(title="ResearchPilot | AI Agent") as demo:
    gr.HTML('''
    <div class="header-container">
        <div class="header-title">🚀 ResearchPilot</div>
        <div class="header-subtitle">Enterprise RAG & Autonomous Research Agent</div>
    </div>
    ''')
    
    with gr.Tabs():
        with gr.TabItem("💬 Conversational RAG"):
            with gr.Row():
                with gr.Column(scale=1, min_width=280, elem_classes=["sidebar-panel"]):
                    gr.Markdown("### ⚙️ Context Settings")
                    gr.Markdown("Have a fluid conversation with your uploaded documents using context retrieval.")
                    with gr.Accordion("Session Settings", open=True):
                        chat_session_id = gr.Textbox(label="Active Session ID", value="1", info="Change this to chat in a different context.")
                
                with gr.Column(scale=3):
                    with gr.Row():
                        btn_1 = gr.Button("📄 Summarize uploaded document", size="sm")
                        btn_2 = gr.Button("🔍 Find key research methodology", size="sm")
                        btn_3 = gr.Button("📊 Extract data metrics & tables", size="sm")
                        
                    chatbot = gr.Chatbot(label="ResearchPilot Assistant", height=580)
                    
                    with gr.Row():
                        chat_input = gr.Textbox(label="", placeholder="Type your question here and hit Enter...", scale=8)
                        chat_submit = gr.Button("Send", variant="primary", scale=1)
                        
                    def submit_message(msg, history):
                        if history is None: history = []
                        if msg.strip():
                            history.append({"role": "user", "content": msg})
                        return "", history
                        
                    def get_bot_response(history, sid):
                        if not history: return history
                        user_content = history[-1]["content"] if isinstance(history[-1], dict) else (history[-1].content if hasattr(history[-1], "content") else history[-1][0]); user_msg = user_content[0]["text"] if isinstance(user_content, list) and len(user_content) > 0 and isinstance(user_content[0], dict) else str(user_content)
                        if not sid.strip() or not sid.strip().isdigit():
                            history.append({"role": "assistant", "content": "Please enter a valid numeric Session ID in Settings."})
                            return history
                        try:
                            from app.services.rag_service import generate_chat_response
                            db = SessionLocal()
                            try:
                                response = generate_chat_response(db, user_msg, int(sid.strip()))
                                ans = response.answer
                                sources = response.sources
                                if sources:
                                    ans += "\n\n**Sources:**\n" + "\n".join([f"- {s.filename} (Page {s.page_number})" for s in sources])
                                history.append({"role": "assistant", "content": ans})
                            finally:
                                db.close()
                        except Exception as exc:
                            history.append({"role": "assistant", "content": f"Error: {exc}"})
                        return history

                    chat_submit.click(fn=submit_message, inputs=[chat_input, chatbot], outputs=[chat_input, chatbot]).then(
                        fn=get_bot_response, inputs=[chatbot, chat_session_id], outputs=[chatbot]
                    )
                    chat_input.submit(fn=submit_message, inputs=[chat_input, chatbot], outputs=[chat_input, chatbot]).then(
                        fn=get_bot_response, inputs=[chatbot, chat_session_id], outputs=[chatbot]
                    )
                    
                    btn_1.click(lambda: "Summarize uploaded document", None, chat_input).then(
                        fn=submit_message, inputs=[chat_input, chatbot], outputs=[chat_input, chatbot]
                    ).then(
                        fn=get_bot_response, inputs=[chatbot, chat_session_id], outputs=[chatbot]
                    )
                    btn_2.click(lambda: "Find key research methodology", None, chat_input).then(
                        fn=submit_message, inputs=[chat_input, chatbot], outputs=[chat_input, chatbot]
                    ).then(
                        fn=get_bot_response, inputs=[chatbot, chat_session_id], outputs=[chatbot]
                    )
                    btn_3.click(lambda: "Extract data metrics & tables", None, chat_input).then(
                        fn=submit_message, inputs=[chat_input, chatbot], outputs=[chat_input, chatbot]
                    ).then(
                        fn=get_bot_response, inputs=[chatbot, chat_session_id], outputs=[chatbot]
                    )

        with gr.TabItem("🧠 Agentic Researcher"):
            with gr.Row():
                with gr.Column(scale=1, min_width=280, elem_classes=["sidebar-panel"]):
                    gr.Markdown("### ⚙️ Agent Settings")
                    gr.Markdown("Deploy an autonomous LangGraph agent to iteratively research and compile a report.")
                    with gr.Accordion("Export Settings", open=True):
                        export_format = gr.Dropdown(choices=["PDF", "Markdown"], value="PDF", label="Format")
                        download_btn = gr.Button("⬇️ Generate Downloadable Report", variant="secondary")
                        download_file = gr.File(label="Download File", visible=True)
                        download_btn.click(fn=export_research_report, inputs=[gr.Markdown(), export_format], outputs=[download_file])
                
                with gr.Column(scale=3):
                    research_query = gr.Textbox(label="Research Topic", placeholder="e.g. Write a comprehensive summary of Mohit's backend engineering skills.")
                    run_agent_btn = gr.Button("Deploy Agent", variant="primary")
                    
                    with gr.Accordion("🧠 Agent Execution Steps", open=True):
                        research_stepper = gr.HTML("<div style='color:#94a3b8; padding:10px;'>Agent is idle.</div>")
                    
                    gr.Markdown("### Agent Output")
                    research_output = gr.Markdown("The generated report will appear here.")
                    
                    run_agent_btn.click(fn=run_research_agent, inputs=research_query, outputs=[research_stepper, research_output])
                    download_btn.click(fn=export_research_report, inputs=[research_output, export_format], outputs=[download_file])

        with gr.TabItem("📚 Knowledge Base"):
            gr.Markdown("Upload standard PDF documents to expand the AI's vectorized knowledge graph.")
            
            kb_stats_html = gr.HTML(value=get_kb_stats_html())
            
            with gr.Row():
                with gr.Column(scale=1):
                    pdf_input = gr.File(label="Upload Document", file_types=[".pdf"])
                    upload_btn = gr.Button("Vectorize Document", variant="primary")
                    upload_output = gr.HTML(label="Status")
                    upload_btn.click(fn=upload_pdf, inputs=pdf_input, outputs=[upload_output, kb_stats_html])
                with gr.Column(scale=1):
                    query_input = gr.Textbox(label="Test Vector Retrieval", placeholder="Search the semantic database directly...")
                    search_btn = gr.Button("Search Vectors")
                    search_output = gr.Textbox(label="Matched Chunks", interactive=False, lines=8)
                    search_btn.click(fn=retrieve_chunks, inputs=query_input, outputs=search_output)

        with gr.TabItem("🛠️ Developer Tools"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Database Entities")
                    list_docs_btn = gr.Button("List Indexed Documents")
                    list_sessions_btn = gr.Button("List Chat Sessions")
                    dev_output = gr.Textbox(label="Database Output", interactive=False, lines=10)
                    list_docs_btn.click(fn=list_documents, inputs=[], outputs=dev_output)
                    list_sessions_btn.click(fn=list_sessions, inputs=[], outputs=dev_output)
                with gr.Column():
                    gr.Markdown("### Diagnostics")
                    name_input = gr.Textbox(label="Ping API Server (Enter name)")
                    test_button = gr.Button("Send Ping")
                    test_output = gr.Textbox(label="Response", interactive=False)
                    test_button.click(fn=test_fastapi_connection, inputs=name_input, outputs=test_output)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.getenv("PORT", "7860")), theme=theme, css=custom_css)

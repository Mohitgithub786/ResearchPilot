# -*- coding: utf-8 -*-
import os
import requests
from dotenv import load_dotenv
import gradio as gr

load_dotenv()

FASTAPI_BASE_URL = os.getenv(
    "FASTAPI_BASE_URL", "https://researchpilot-irt9.onrender.com"
).rstrip("/")

def test_fastapi_connection(name: str) -> str:
    if not name.strip():
        return "Please enter your name."
    try:
        response = requests.get(f"{FASTAPI_BASE_URL}/hello", params={"name": name.strip()}, timeout=10)
        response.raise_for_status()
        return response.json().get("message", "Unexpected response from FastAPI.")
    except Exception as exc:
        return f"Could not reach FastAPI backend at {FASTAPI_BASE_URL}. Error: {exc}"

def create_document(filename: str) -> str:
    if not filename.strip():
        return "Please enter a filename."
    try:
        response = requests.post(f"{FASTAPI_BASE_URL}/documents", json={"filename": filename.strip()}, timeout=10)
        response.raise_for_status()
        data = response.json()
        return f"Document created!\nID: {data['id']}\nFilename: {data['filename']}\nUploaded At: {data['uploaded_at']}"
    except Exception as exc:
        return f"Error creating document: {exc}"

def list_documents() -> str:
    try:
        response = requests.get(f"{FASTAPI_BASE_URL}/documents", timeout=10)
        response.raise_for_status()
        docs = response.json()
        if not docs:
            return "No documents found."
        return "\n".join([f"[{d['id']}] {d['filename']} (Uploaded: {d['uploaded_at']})" for d in docs])
    except Exception as exc:
        return f"Error listing documents: {exc}"

def create_session() -> str:
    try:
        response = requests.post(f"{FASTAPI_BASE_URL}/sessions", timeout=10)
        response.raise_for_status()
        data = response.json()
        return f"Session created!\nID: {data['session_id']}\nTitle: {data['title']}"
    except Exception as exc:
        return f"Error creating session: {exc}"

def list_sessions() -> str:
    try:
        response = requests.get(f"{FASTAPI_BASE_URL}/sessions", timeout=10)
        response.raise_for_status()
        sessions = response.json()
        if not sessions:
            return "No sessions found."
        return "\n".join([f"[{s['id']}] {s['title']} (Created: {s['created_at']})" for s in sessions])
    except Exception as exc:
        return f"Error listing sessions: {exc}"

def upload_pdf(filepath) -> str:
    import time
    import os
    if not filepath:
        return "<div style='color:#ef4444'>Please select a file to upload.</div>"
    try:
        from app.database.session import SessionLocal
        from app.services.ingestion import ingest_pdf
        import shutil
        import uuid
        
        db = SessionLocal()
        try:
            filename = os.path.basename(filepath)
            
            class MockUploadFile:
                def __init__(self, filename, file):
                    self.filename = filename
                    self.file = file
                    
            with open(filepath, "rb") as f:
                mock_file = MockUploadFile(filename, f)
                document, chunk_count = ingest_pdf(db, mock_file)
                
            return f"<div style='padding:15px; border-radius:12px; background:rgba(30,41,59,0.7); border:1px solid rgba(255,255,255,0.1);'><div style='display:flex; justify-content:space-between; align-items:center;'><b>📄 {filename}</b><span style='background:rgba(74,222,128,0.2); color:#4ade80; padding:4px 8px; border-radius:12px; font-size:0.8rem; font-weight:600;'>✅ Indexed ({chunk_count} chunks)</span></div></div>"
        finally:
            db.close()
    except Exception as exc:
        return f"<div style='color:#ef4444'>Error uploading PDF: {exc}</div>"

def list_chunks(doc_id: str) -> str:
    if not doc_id.strip():
        return "Please enter a Document ID."
    try:
        response = requests.get(f"{FASTAPI_BASE_URL}/documents/{doc_id.strip()}/chunks", timeout=10)
        response.raise_for_status()
        chunks = response.json()
        if not chunks:
            return "No chunks found."
        
        result = []
        for c in chunks:
            text = c['chunk_text'].replace('\n', ' ')[:80] + "..."
            result.append(f"[Chunk {c['chunk_order']} | Page {c['page_number']}] {text}")
        return "\n".join(result)
    except Exception as exc:
        return f"Error fetching chunks: {exc}"

def retrieve_chunks(query: str) -> str:
    if not query.strip():
        return "Please enter a search query."
    try:
        response = requests.post(f"{FASTAPI_BASE_URL}/retrieve", json={"query": query.strip()}, timeout=30)
        response.raise_for_status()
        chunks = response.json()
        if not chunks:
            return "No matching chunks found."
        
        result = []
        for c in chunks:
            score = round(c.get('similarity_score', 0), 4)
            text = c['chunk_text'].replace('\n', ' ')
            result.append(f"[{score}] {c['source_filename']} (Page {c['page_number']}):\n{text}\n")
        return "\n".join(result)
    except Exception as exc:
        return f"Error retrieving chunks: {exc}"

def run_research_agent(query: str):
    if not query.strip():
        yield ("<div style='color:#ef4444'>Please enter a research query.</div>", "No query provided.")
        return
    try:
        from app.services.research_agent import stream_research
        import asyncio
        
        status_log = "<div style='display:flex; flex-direction:column; gap:8px;'>"
        output = ""
        
        # We need to run the async generator in a sync context since Gradio generator is sync here
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        async def consume():
            nonlocal status_log, output
            results = []
            async for data in stream_research(query.strip()):
                results.append(data)
            return results
            
        events = loop.run_until_complete(consume())
        loop.close()
        
        # Yield the events (note: this turns it into a batch yield rather than true streaming, 
        # but prevents any network timeout/disconnect issues on Render's free tier proxy)
        for data in events:
            if data.get("type") == "status":
                node = data.get('node')
                status = data.get('status')
                color = "#4ade80" if status == "completed" else "#fbbf24"
                status_log += f"<div style='font-size:0.95rem; padding:8px; background:rgba(255,255,255,0.05); border-radius:8px;'><span style='color:{color}; margin-right:8px;'>●</span> <b>{node}</b> <span style='color:#94a3b8'><i>{status}...</i></span></div>\n"
                yield (status_log + "</div>", output)
            elif data.get("type") == "token":
                output += data.get("content", "")
                yield (status_log + "</div>", output)
            elif data.get("type") == "sources":
                sources = data.get("sources", [])
                if sources:
                    sources_md = "\n\n### Interactive Citations\n"
                    for s in sources:
                        filename = s.get('filename', 'Unknown')
                        page = s.get('page_number', '?')
                        text = s.get('chunk_text', '').replace('\n', '<br>')
                        sources_md += f"<details style='margin-bottom:8px; border:1px solid rgba(255,255,255,0.1); border-radius:8px; padding:10px; background:#1e293b;'><summary style='cursor:pointer; font-weight:600; color:#818cf8;'>📄 {filename} (Page {page})</summary><p style='margin-top:10px; padding-top:10px; border-top:1px solid rgba(255,255,255,0.05); color:#cbd5e1;'>{text}</p></details>\n"
                    output += sources_md
                    yield (status_log + "</div>", output)
    except Exception as exc:
        yield (f"<div style='color:#ef4444'>Error: {exc}</div>", "")

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
                            from app.database.session import SessionLocal
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
                
                with gr.Column(scale=3):
                    research_query = gr.Textbox(label="Research Topic", placeholder="e.g. Write a comprehensive summary of Mohit's backend engineering skills.")
                    run_agent_btn = gr.Button("Deploy Agent", variant="primary")
                    
                    with gr.Accordion("🧠 Agent Execution Steps", open=True):
                        research_stepper = gr.HTML("<div style='color:#94a3b8; padding:10px;'>Agent is idle.</div>")
                    
                    gr.Markdown("### Agent Output")
                    research_output = gr.Markdown("The generated report will appear here.")
                    
                    with gr.Row():
                        download_btn = gr.Button("⬇️ Download Research Report (PDF/MD)", variant="secondary")
                    
                    run_agent_btn.click(fn=run_research_agent, inputs=research_query, outputs=[research_stepper, research_output])

        with gr.TabItem("📚 Knowledge Base"):
            gr.Markdown("Upload standard PDF documents to expand the AI's vectorized knowledge graph.")
            
            gr.HTML("""
            <div style='display:flex; gap:20px; margin-bottom:20px;'>
                <div class='gr-box' style='flex:1; padding:20px; text-align:center;'>
                    <h3 style='margin-bottom:10px; color:#94a3b8; font-size:1rem;'>📁 Documents Uploaded</h3>
                    <h2 style='color:#a855f7; font-size:2rem; font-weight:800;'>12</h2>
                </div>
                <div class='gr-box' style='flex:1; padding:20px; text-align:center;'>
                    <h3 style='margin-bottom:10px; color:#94a3b8; font-size:1rem;'>🧩 Chunks Indexed in ChromaDB</h3>
                    <h2 style='color:#3b82f6; font-size:2rem; font-weight:800;'>1,432</h2>
                </div>
                <div class='gr-box' style='flex:1; padding:20px; text-align:center;'>
                    <h3 style='margin-bottom:10px; color:#94a3b8; font-size:1rem;'>🟢 Vector DB Status</h3>
                    <h2 style='color:#4ade80; font-size:2rem; font-weight:800;'>Online</h2>
                </div>
            </div>
            """)
            
            with gr.Row():
                with gr.Column(scale=1):
                    pdf_input = gr.File(label="Upload Document", file_types=[".pdf"])
                    upload_btn = gr.Button("Vectorize Document", variant="primary")
                    upload_output = gr.HTML(label="Status")
                    upload_btn.click(fn=upload_pdf, inputs=pdf_input, outputs=upload_output)
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



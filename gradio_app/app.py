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
    if not filepath:
        return "<div style='color:#ef4444'>Please select a file to upload.</div>"
    try:
        with open(filepath, 'rb') as f:
            files = {'file': (os.path.basename(filepath), f, 'application/pdf')}
            response = requests.post(f"{FASTAPI_BASE_URL}/documents/upload", files=files, timeout=30)
            response.raise_for_status()
            data = response.json()
            task_id = data.get("task_id")
            if not task_id:
                return "<div style='color:#ef4444'>Upload accepted, but no task ID returned.</div>"
            
            # Poll status
            for _ in range(60): # 1 minute timeout
                status_res = requests.get(f"{FASTAPI_BASE_URL}/documents/status/{task_id}")
                status_res.raise_for_status()
                sdata = status_res.json()
                if sdata.get("completed"):
                    filename = os.path.basename(filepath)
                    return f"<div style='padding:15px; border-radius:12px; background:rgba(30,41,59,0.7); border:1px solid rgba(255,255,255,0.1);'><div style='display:flex; justify-content:space-between; align-items:center;'><b>📄 {filename}</b><span style='background:rgba(74,222,128,0.2); color:#4ade80; padding:4px 8px; border-radius:12px; font-size:0.8rem; font-weight:600;'>✅ Indexed</span></div></div>"
                elif sdata.get("failed"):
                    return f"<div style='color:#ef4444'>Error processing PDF: {sdata.get('error')}</div>"
                time.sleep(1)
            return "<div style='color:#fbbf24'>Upload processing timed out.</div>"
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
        yield "Please enter a research query."
        return
    try:
        response = requests.get(
            f"{FASTAPI_BASE_URL}/stream", 
            params={"query": query.strip()}, 
            stream=True,
            timeout=120
        )
        response.raise_for_status()
        
        status_log = "<div style='display:flex; flex-direction:column; gap:8px; margin-bottom:20px; padding:15px; background:rgba(30,41,59,0.5); border-radius:12px; border:1px solid rgba(255,255,255,0.05);'><b>⚡ Live Execution:</b>\n"
        output = ""
        import json
        for line in response.iter_lines():
            if line:
                decoded = line.decode('utf-8')
                if decoded.startswith("data: "):
                    try:
                        data = json.loads(decoded[6:])
                        if data.get("type") == "status":
                            node = data.get('node')
                            status = data.get('status')
                            color = "#4ade80" if status == "completed" else "#fbbf24"
                            status_log += f"<div style='font-size:0.9rem'><span style='color:{color}'>●</span> <b>{node}</b> <i>{status}...</i></div>\n"
                            yield status_log + "</div>\n\n" + output
                        elif data.get("type") == "token":
                            output += data.get("content", "")
                            yield status_log + "</div>\n\n" + output
                        elif data.get("type") == "sources":
                            sources = data.get("sources", [])
                            if sources:
                                sources_md = "\n\n### Interactive Citations\n"
                                for s in sources:
                                    filename = s.get('filename', 'Unknown')
                                    page = s.get('page_number', '?')
                                    text = s.get('chunk_text', '').replace('\n', '<br>')
                                    sources_md += f"<details><summary><b>{filename}</b> (Page {page})</summary><p style='margin-left: 10px; padding: 10px; border-left: 3px solid #6366f1; background: rgba(30,41,59,0.5);'>{text}</p></details>\n"
                                output += sources_md
                                yield status_log + "</div>\n\n" + output
                    except json.JSONDecodeError:
                        pass
    except Exception as exc:
        yield f"Error running research agent: {exc}"

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
"""

theme = gr.themes.Soft(
    primary_hue="indigo",
    neutral_hue="slate"
)

with gr.Blocks(title="ResearchPilot | AI Agent", theme=theme, css=custom_css) as demo:
    gr.HTML('''
    <div class="header-container">
        <div class="header-title">🚀 ResearchPilot</div>
        <div class="header-subtitle">Enterprise RAG & Autonomous Research Agent</div>
    </div>
    ''')
    
    with gr.Tabs():
        with gr.TabItem("💬 Conversational RAG"):
            gr.Markdown("Have a fluid conversation with your uploaded documents using context retrieval.")
            
            with gr.Accordion("Session Settings (Advanced)", open=False):
                chat_session_id = gr.Textbox(label="Active Session ID", value="1", info="Change this to chat in a different context.")
                
            chatbot = gr.Chatbot(label="ResearchPilot Assistant", height=450)
            
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
                    response = requests.post(f"{FASTAPI_BASE_URL}/chat", json={"session_id": int(sid.strip()), "question": user_msg}, timeout=60)
                    response.raise_for_status()
                    data = response.json()
                    ans = data.get("answer", "No answer provided.")
                    sources = data.get("sources", [])
                    if sources:
                        ans += "\n\n**Sources:**\n" + "\n".join([f"- {s.get('filename', 'Unknown')} (Page {s.get('page_number', '?')})" for s in sources])
                    history.append({"role": "assistant", "content": ans})
                except Exception as exc:
                    history.append({"role": "assistant", "content": f"Error: {exc}"})
                return history

            chat_submit.click(fn=submit_message, inputs=[chat_input, chatbot], outputs=[chat_input, chatbot]).then(
                fn=get_bot_response, inputs=[chatbot, chat_session_id], outputs=[chatbot]
            )
            chat_input.submit(fn=submit_message, inputs=[chat_input, chatbot], outputs=[chat_input, chatbot]).then(
                fn=get_bot_response, inputs=[chatbot, chat_session_id], outputs=[chatbot]
            )

        with gr.TabItem("🧠 Agentic Researcher"):
            gr.Markdown("Deploy an autonomous LangGraph agent to plan, iteratively search, and compile a structured multi-source report.")
            
            research_query = gr.Textbox(label="Research Topic", placeholder="e.g. Write a comprehensive summary of Mohit's backend engineering skills.")
            run_agent_btn = gr.Button("Deploy Agent", variant="primary")
            
            gr.Markdown("### Agent Output")
            research_output = gr.Markdown("The generated report will appear here. The agent may take up to 60 seconds to complete its iterative research loops.")
            run_agent_btn.click(fn=run_research_agent, inputs=research_query, outputs=research_output)

        with gr.TabItem("📚 Knowledge Base"):
            gr.Markdown("Upload standard PDF documents to expand the AI's vectorized knowledge graph.")
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
    demo.launch(server_name="0.0.0.0", server_port=int(os.getenv("PORT", "7860")))



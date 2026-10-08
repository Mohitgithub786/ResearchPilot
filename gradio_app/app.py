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
    if not filepath:
        return "Please select a file to upload."
    try:
        with open(filepath, 'rb') as f:
            files = {'file': (os.path.basename(filepath), f, 'application/pdf')}
            response = requests.post(f"{FASTAPI_BASE_URL}/documents/upload", files=files, timeout=30)
            response.raise_for_status()
            data = response.json()
            return f"Upload successful!\nDocument ID: {data['id']}\nFilename: {data['filename']}\nChunks Created: {data['chunk_count']}"
    except Exception as exc:
        return f"Error uploading PDF: {exc}"

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

def run_research_agent(query: str) -> str:
    if not query.strip():
        return "Please enter a research query."
    try:
        response = requests.post(
            f"{FASTAPI_BASE_URL}/research", 
            json={"query": query.strip()}, 
            timeout=120
        )
        response.raise_for_status()
        data = response.json()
        
        summary = data.get("summary", "")
        key_findings = data.get("key_findings", [])
        sources = data.get("sources", [])
        
        result = f"### Summary\n{summary}\n\n### Key Findings\n"
        for finding in key_findings:
            result += f"- {finding}\n"
            
        if sources:
            result += "\n### Sources\n"
            for s in sources:
                result += f"- {s.get('filename', 'Unknown')} (Page {s.get('page_number', '?')})\n"
                
        return result
    except Exception as exc:
        return f"Error running research agent: {exc}"

custom_css = """
footer {display: none !important;}
.gradio-container {max-width: 1000px !important;}
.main-header {text-align: center; margin-bottom: 1rem; margin-top: 1rem;}
.main-header h1 {font-size: 2.8rem; color: #0f172a; font-weight: 800; margin-bottom: 0.2rem;}
.main-header p {color: #475569; font-size: 1.1rem;}
"""

theme = gr.themes.Soft(
    primary_hue="indigo",
    secondary_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
).set(
    button_primary_background_fill="*primary_600",
    button_primary_background_fill_hover="*primary_700",
    block_radius="lg",
)

with gr.Blocks(title="ResearchPilot | AI Agent", theme=theme, css=custom_css) as demo:
    gr.HTML('''
    <div class="main-header">
        <h1>?? ResearchPilot</h1>
        <p>Enterprise RAG & Autonomous Research Agent</p>
    </div>
    ''')
    
    with gr.Tabs():
        with gr.TabItem("?? Conversational RAG"):
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

        with gr.TabItem("?? Agentic Researcher"):
            gr.Markdown("Deploy an autonomous LangGraph agent to plan, iteratively search, and compile a structured multi-source report.")
            
            research_query = gr.Textbox(label="Research Topic", placeholder="e.g. Write a comprehensive summary of Mohit's backend engineering skills.")
            run_agent_btn = gr.Button("Deploy Agent", variant="primary")
            
            gr.Markdown("### Agent Output")
            research_output = gr.Markdown("The generated report will appear here. The agent may take up to 60 seconds to complete its iterative research loops.")
            run_agent_btn.click(fn=run_research_agent, inputs=research_query, outputs=research_output)

        with gr.TabItem("?? Knowledge Base"):
            gr.Markdown("Upload standard PDF documents to expand the AI's vectorized knowledge graph.")
            with gr.Row():
                with gr.Column(scale=1):
                    pdf_input = gr.File(label="Upload Document", file_types=[".pdf"])
                    upload_btn = gr.Button("Vectorize Document", variant="primary")
                    upload_output = gr.Textbox(label="Status", interactive=False, lines=4)
                    upload_btn.click(fn=upload_pdf, inputs=pdf_input, outputs=upload_output)
                with gr.Column(scale=1):
                    query_input = gr.Textbox(label="Test Vector Retrieval", placeholder="Search the semantic database directly...")
                    search_btn = gr.Button("Search Vectors")
                    search_output = gr.Textbox(label="Matched Chunks", interactive=False, lines=8)
                    search_btn.click(fn=retrieve_chunks, inputs=query_input, outputs=search_output)

        with gr.TabItem("?? Developer Tools"):
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

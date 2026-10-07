import os
import requests
from dotenv import load_dotenv
import gradio as gr

load_dotenv()

FASTAPI_BASE_URL = os.getenv(
    "FASTAPI_BASE_URL", "http://127.0.0.1:8000"
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
        return "\n".join([f"[{s['session_id']}] {s['title']} (Created: {s['created_at']})" for s in sessions])
    except Exception as exc:
        return f"Error listing sessions: {exc}"

with gr.Blocks(title="ResearchPilot") as demo:
    gr.Markdown("# ResearchPilot")
    
    with gr.Tab("Phase 2: Test Connection"):
        gr.Markdown("Test the connection to the FastAPI backend.")
        name_input = gr.Textbox(label="Enter your name")
        test_button = gr.Button("Test FastAPI Connection")
        test_output = gr.Textbox(label="Response", interactive=False)
        test_button.click(fn=test_fastapi_connection, inputs=name_input, outputs=test_output)
        
    with gr.Tab("Phase 3: Persistence"):
        gr.Markdown("Test the SQLite persistence for Documents and Chat Sessions.")
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Documents")
                doc_filename = gr.Textbox(label="Document Filename")
                create_doc_btn = gr.Button("Create Document Record")
                list_docs_btn = gr.Button("List All Documents")
                doc_output = gr.Textbox(label="Document Output", interactive=False, lines=5)
                
                create_doc_btn.click(fn=create_document, inputs=doc_filename, outputs=doc_output)
                list_docs_btn.click(fn=list_documents, inputs=[], outputs=doc_output)
                
            with gr.Column():
                gr.Markdown("### Sessions")
                create_session_btn = gr.Button("Create Chat Session")
                list_sessions_btn = gr.Button("List All Sessions")
                session_output = gr.Textbox(label="Session Output", interactive=False, lines=5)
                
                create_session_btn.click(fn=create_session, inputs=[], outputs=session_output)
                list_sessions_btn.click(fn=list_sessions, inputs=[], outputs=session_output)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.getenv("PORT", "7860")))
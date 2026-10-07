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

    with gr.Tab("Phase 4: Ingestion"):
        gr.Markdown("Upload a real PDF file. The backend will parse it, chunk the text, and store the chunks in the database.")
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Upload PDF")
                pdf_input = gr.File(label="Select PDF File", file_types=[".pdf"])
                upload_btn = gr.Button("Upload & Process PDF")
                upload_output = gr.Textbox(label="Upload Output", interactive=False, lines=4)
                
                upload_btn.click(fn=upload_pdf, inputs=pdf_input, outputs=upload_output)
                
            with gr.Column():
                gr.Markdown("### View Chunks")
                chunk_doc_id = gr.Textbox(label="Enter Document ID (e.g. 1)")
                view_chunks_btn = gr.Button("View Document Chunks")
                chunks_output = gr.Textbox(label="Chunks Preview", interactive=False, lines=10)
                
                view_chunks_btn.click(fn=list_chunks, inputs=chunk_doc_id, outputs=chunks_output)

    with gr.Tab("Phase 5: Retrieval"):
        gr.Markdown("Search across all embedded document chunks using ChromaDB vector similarity.")
        
        with gr.Row():
            with gr.Column():
                query_input = gr.Textbox(label="Search Query", placeholder="e.g. What is positional encoding?")
                search_btn = gr.Button("Search Vectors")
            with gr.Column():
                search_output = gr.Textbox(label="Top Matching Chunks", interactive=False, lines=12)
                
        search_btn.click(fn=retrieve_chunks, inputs=query_input, outputs=search_output)

    with gr.Tab("Phase 6 & 7: Conversational RAG"):
        gr.Markdown("Chat with your uploaded documents using context retrieval and session-based memory.")
        
        chat_session_id = gr.Textbox(label="Session ID (Create one in Phase 3 first!)", value="1")
        chatbot = gr.Chatbot(label="Research Assistant")
        
        with gr.Row():
            chat_input = gr.Textbox(label="Your Message", placeholder="Type your question here...")
            chat_submit = gr.Button("Send")
            
        def submit_message(msg, history):
            return "", history + [[msg, None]]
            
        def get_bot_response(history, sid):
            if not history:
                return history
            user_msg = history[-1][0]
            
            if not sid.strip() or not sid.strip().isdigit():
                history[-1][1] = "Please enter a valid numeric Session ID."
                return history
                
            try:
                response = requests.post(
                    f"{FASTAPI_BASE_URL}/chat", 
                    json={"session_id": int(sid.strip()), "question": user_msg}, 
                    timeout=60
                )
                response.raise_for_status()
                data = response.json()
                
                ans = data.get("answer", "No answer provided.")
                sources = data.get("sources", [])
                
                if sources:
                    src_text = "\n\n**Sources:**\n" + "\n".join([f"- {s.get('filename', 'Unknown')} (Page {s.get('page_number', '?')})" for s in sources])
                    ans += src_text
                    
                history[-1][1] = ans
            except Exception as exc:
                history[-1][1] = f"Error: {exc}"
            
            return history

        chat_submit.click(fn=submit_message, inputs=[chat_input, chatbot], outputs=[chat_input, chatbot]).then(
            fn=get_bot_response, inputs=[chatbot, chat_session_id], outputs=[chatbot]
        )
        chat_input.submit(fn=submit_message, inputs=[chat_input, chatbot], outputs=[chat_input, chatbot]).then(
            fn=get_bot_response, inputs=[chatbot, chat_session_id], outputs=[chatbot]
        )

    with gr.Tab("Phase 8: Agentic Research"):
        gr.Markdown("Run a multi-step LangGraph agent to plan, retrieve, and write a structured research report.")
        
        with gr.Row():
            with gr.Column():
                research_query = gr.Textbox(label="Research Topic", placeholder="e.g. Compare positional encoding methods.")
                run_agent_btn = gr.Button("Run Agentic Research")
            with gr.Column():
                research_output = gr.Markdown("Report will appear here...")
                
        run_agent_btn.click(fn=run_research_agent, inputs=research_query, outputs=research_output)

    with gr.Tab("Phase 9: Observability"):
        gr.Markdown("### Logging & Streaming")
        gr.Markdown("Phase 9 focuses on backend observability. The FastAPI server uses structured logging to track agent steps, database queries, and vector searches. Check your terminal running `start.bat` to see these detailed logs in real-time while you use the other tabs!")

    with gr.Tab("Phase 10: Deployment"):
        gr.Markdown("### Dockerization & CI/CD")
        gr.Markdown("The project is now fully production-ready (Final Phase).")
        gr.Markdown("To run this in a production environment (like AWS or Render), you can use the included Docker setup:")
        gr.Code("docker-compose up --build", language="shell")
        gr.Markdown("This spins up the FastAPI backend and ChromaDB containerized environments seamlessly.")

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.getenv("PORT", "7860")))
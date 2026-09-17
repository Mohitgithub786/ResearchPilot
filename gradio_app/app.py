"""
ResearchPilot Gradio UI.

Gradio runs as a separate process from the FastAPI backend. The UI sends HTTP
requests to FastAPI over the network (default: http://127.0.0.1:8000), rather
than importing backend code directly.

Why separate UI and backend?
- Each service can be developed, scaled, and deployed independently.
- The same FastAPI API can serve Gradio, mobile clients, or other integrations.
- Future GenAI workloads (RAG, agents, streaming) stay on the backend while
  the UI remains a thin client that calls HTTP endpoints.

This pattern will support later phases: document upload, chat, and streaming
responses will all flow through FastAPI endpoints that Gradio consumes.
"""

import json
import os
import urllib.error
import urllib.parse
import urllib.request

import gradio as gr

# Default used when FASTAPI_BASE_URL is not set in the environment.
DEFAULT_FASTAPI_BASE_URL = "http://127.0.0.1:8000"

# Configurable via FASTAPI_BASE_URL (see .env.example). Gradio calls this base
# URL over HTTP; it does not import FastAPI application code directly.
FASTAPI_BASE_URL = os.getenv("FASTAPI_BASE_URL", DEFAULT_FASTAPI_BASE_URL).rstrip("/")


def test_fastapi_connection(name: str) -> str:
    """
    Call FastAPI GET /hello?name=<name> and return the message from the JSON body.

    Gradio invokes this handler when the user clicks "Test FastAPI Connection".
    The handler builds an HTTP request, sends it to the running FastAPI process,
    parses the JSON response, and displays the result in the output textbox.
    """
    if not name.strip():
        return "Please enter your name."

    params = urllib.parse.urlencode({"name": name.strip()})
    url = f"{FASTAPI_BASE_URL}/hello?{params}"

    try:
        with urllib.request.urlopen(url, timeout=10) as response:
            data = json.loads(response.read().decode())
            return data.get("message", "Unexpected response from FastAPI.")
    except urllib.error.URLError as exc:
        return (
            f"Could not reach FastAPI at {FASTAPI_BASE_URL}. "
            f"Make sure the server is running (uvicorn app.main:app --reload). "
            f"Error: {exc.reason}"
        )


with gr.Blocks(title="ResearchPilot") as demo:
    gr.Markdown("# ResearchPilot")
    gr.Markdown(
        "Phase 2: Enter your name and test the connection to the FastAPI backend."
    )

    name_input = gr.Textbox(label="Enter your name")
    test_button = gr.Button("Test FastAPI Connection")
    output = gr.Textbox(label="Response", interactive=False)

    test_button.click(
        fn=test_fastapi_connection,
        inputs=name_input,
        outputs=output,
    )

if __name__ == "__main__":
    demo.launch()

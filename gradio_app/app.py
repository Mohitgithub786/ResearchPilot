"""
ResearchPilot Gradio UI.

Gradio runs as a separate process from the FastAPI backend. The UI sends HTTP
requests to FastAPI over the network, rather than importing backend code directly.
"""

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from dotenv import load_dotenv

load_dotenv()

import gradio as gr

# Render Backend Production URL as default fallback
FASTAPI_BASE_URL = os.getenv(
    "FASTAPI_BASE_URL", "https://researchpilot-backend-a965.onrender.com"
).rstrip("/")


def test_fastapi_connection(name: str) -> str:
    """Call FastAPI GET /hello?name=<name> and return the message from the JSON body."""
    if not name.strip():
        return "Please enter your name."

    params = urllib.parse.urlencode({"name": name.strip()})
    url = f"{FASTAPI_BASE_URL}/hello?{params}"

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=10) as response:
            data = json.loads(response.read().decode())
            return data.get("message", "Unexpected response from FastAPI.")
    except urllib.error.URLError as exc:
        return (
            f"Could not reach FastAPI backend at {FASTAPI_BASE_URL}. "
            f"Error details: {exc.reason}"
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
    demo.launch(server_name="0.0.0.0", server_port=int(os.getenv("PORT", "7860")))
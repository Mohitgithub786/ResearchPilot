from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_document_not_found_error_handler():
    # Calling a non-existent document chunks endpoint should trigger DocumentNotFoundError handler
    response = client.get("/documents/999999/chunks")
    assert response.status_code == 404
    assert response.json() == {"error": "Document with ID 999999 not found."}

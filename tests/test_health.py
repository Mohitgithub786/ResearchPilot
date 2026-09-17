from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "database" in data
    assert "chroma" in data
    assert "embedding_provider" in data
    # Depending on test environment, it could be connected or disconnected,
    # but the structure must be present.
    assert data["status"] in ["healthy", "unhealthy"]

import uuid

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database.base import Base
from app.database.session import get_db
from app.main import app as fastapi_app
from app.services.chroma_service import reset_chroma_client
from app.services import embedding_service
from app.services.embedding_service import get_embedding_model
from app.services.llm_service import get_llm_service


@pytest.fixture()
def client(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    upload_dir = data_dir / "uploads"
    chroma_dir = data_dir / "chroma_db"
    upload_dir.mkdir(parents=True)
    chroma_dir.mkdir(parents=True)

    db_path = data_dir / "test.db"
    engine = create_engine(
        f"sqlite:///{db_path}",
        connect_args={"check_same_thread": False},
    )
    testing_session_factory = sessionmaker(
        autocommit=False, autoflush=False, bind=engine
    )

    import app.models.chat_session  # noqa: F401
    import app.models.chunk  # noqa: F401
    import app.models.document  # noqa: F401
    import app.models.message  # noqa: F401

    Base.metadata.create_all(bind=engine)

    collection_name = f"test_{uuid.uuid4().hex}"
    monkeypatch.setenv("EMBEDDING_PROVIDER", "test")
    monkeypatch.setenv("LLM_PROVIDER", "test")
    monkeypatch.setenv("CHROMA_COLLECTION_NAME", collection_name)
    monkeypatch.setattr("app.core.config.DATA_DIR", data_dir)
    monkeypatch.setattr("app.core.config.UPLOAD_DIR", upload_dir)
    monkeypatch.setattr("app.core.config.CHROMA_DIR", chroma_dir)
    monkeypatch.setattr("app.core.config.CHROMA_COLLECTION_NAME", collection_name)
    reset_chroma_client()
    # Clear the embedding model singleton and dimension cache so the test
    # provider (test stub) is used with its own vector size
    get_embedding_model.cache_clear()
    get_llm_service.cache_clear()
    embedding_service._cached_dim = None

    def override_get_db():
        db = testing_session_factory()
        try:
            yield db
        finally:
            db.close()

    fastapi_app.dependency_overrides[get_db] = override_get_db

    with TestClient(fastapi_app) as test_client:
        yield test_client, testing_session_factory, chroma_dir

    fastapi_app.dependency_overrides.clear()
    get_embedding_model.cache_clear()
    get_llm_service.cache_clear()
    embedding_service._cached_dim = None
    reset_chroma_client()
    engine.dispose()

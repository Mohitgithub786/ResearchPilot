import os
from pathlib import Path

from dotenv import load_dotenv
from app.core.errors import ConfigurationError

load_dotenv()
DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
CHROMA_DIR = DATA_DIR / "chroma_db"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_DIR.mkdir(parents=True, exist_ok=True)

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))

EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "openai").lower()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_EMBEDDING_MODEL = os.getenv(
    "OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"
)
HF_EMBEDDING_MODEL = os.getenv(
    "HF_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)

CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "researchpilot_chunks")
RETRIEVAL_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "5"))

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq").lower()
OPENAI_LLM_MODEL = os.getenv("OPENAI_LLM_MODEL", "gpt-4o-mini")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

MEMORY_WINDOW = int(os.getenv("MEMORY_WINDOW", "5"))

RESEARCH_RELEVANCE_THRESHOLD = float(
    os.getenv("RESEARCH_RELEVANCE_THRESHOLD", "0.25")
)

def validate_config():
    import os
    if not UPLOAD_DIR.exists():
        raise ConfigurationError(f"Upload directory does not exist: {UPLOAD_DIR}")
    if not CHROMA_DIR.exists():
        raise ConfigurationError(f"Chroma directory does not exist: {CHROMA_DIR}")

    current_embedding = os.getenv("EMBEDDING_PROVIDER", EMBEDDING_PROVIDER).lower()
    current_openai = os.getenv("OPENAI_API_KEY", OPENAI_API_KEY)
    current_llm = os.getenv("LLM_PROVIDER", LLM_PROVIDER).lower()
    current_groq = os.getenv("GROQ_API_KEY", GROQ_API_KEY)

    if current_embedding == "openai" and not current_openai:
        raise ConfigurationError("OPENAI_API_KEY is required when EMBEDDING_PROVIDER is openai")
    
    if current_llm == "groq" and not current_groq:
        raise ConfigurationError("GROQ_API_KEY is required when LLM_PROVIDER is groq")
    
    if current_llm == "openai" and not current_openai:
        raise ConfigurationError("OPENAI_API_KEY is required when LLM_PROVIDER is openai")

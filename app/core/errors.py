class ResearchPilotError(Exception):
    """Base exception for all ResearchPilot custom errors."""
    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


class DocumentNotFoundError(ResearchPilotError):
    """Raised when a requested document is not found."""
    pass


class RetrievalError(ResearchPilotError):
    """Raised when there is an issue retrieving relevant chunks."""
    pass


class EmbeddingError(ResearchPilotError):
    """Raised when there is an issue generating embeddings."""
    pass


class ResearchExecutionError(ResearchPilotError):
    """Raised when the agentic research workflow fails."""
    pass


class ConfigurationError(ResearchPilotError):
    """Raised when there is an invalid or missing configuration."""
    pass

"""
Configurable LLM provider factory.

ResearchPilot uses a small generate(prompt) -> str interface so the RAG service
does not depend on a specific vendor SDK.

Supported providers:
- groq:    ChatGroq via langchain-groq (production default)
- openai:  ChatOpenAI via langchain-openai
- test:    Deterministic stub for pytest (no API calls)

Future providers (add a branch in get_llm_service):
- ollama:  ChatOllama from langchain-ollama (local models)
"""

from app.core.logging import get_logger
import os
from abc import ABC, abstractmethod
from functools import lru_cache

from app.core.config import (
    GROQ_API_KEY,
    GROQ_MODEL,
    LLM_PROVIDER,
    OPENAI_API_KEY,
    OPENAI_LLM_MODEL,
)

logger = get_logger(__name__)


class LLMService(ABC):
    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Generate a text completion from a fully constructed prompt."""


class OpenAILLMService(LLMService):
    def __init__(self, model: str, api_key: str | None) -> None:
        from langchain_openai import ChatOpenAI

        self._llm = ChatOpenAI(model=model, api_key=api_key, temperature=0)

    def generate(self, prompt: str) -> str:
        response = self._llm.invoke(prompt)
        return str(response.content).strip()


class GroqLLMService(LLMService):
    def __init__(self, model: str, api_key: str | None) -> None:
        from langchain_groq import ChatGroq

        self._llm = ChatGroq(model=model, api_key=api_key, temperature=0)

    def generate(self, prompt: str) -> str:
        response = self._llm.invoke(prompt)
        return str(response.content).strip()


class TestLLMService(LLMService):
    """
    Deterministic LLM stub for automated tests.

    Returns grounded-style answers when known context phrases appear in the
    prompt; otherwise returns a generic context-based acknowledgement.
    Supports conversational prompts with history for multi-turn testing.
    """

    def generate(self, prompt: str) -> str:
        prompt_lower = prompt.lower()

        # Multi-turn: detect follow-up about limitations when history
        # mentions self-attention
        if (
            "conversation history:" in prompt_lower
            and "self-attention" in prompt_lower
            and "limitation" in prompt_lower
        ):
            return (
                "The limitations of self-attention include quadratic "
                "computational complexity with sequence length."
            )

        if (
            "positional encoding injects order" in prompt_lower
            or "positional encoding is essential" in prompt_lower
            or "positional encoding is added before" in prompt_lower
        ):
            return (
                "Positional encoding injects order information into "
                "transformer inputs."
            )
        if "sqlite stores structured relational metadata" in prompt_lower:
            return (
                "SQLite stores structured relational metadata for documents."
            )
        if "self-attention computes weighted relationships" in prompt_lower:
            return (
                "Self-attention computes weighted relationships between tokens."
            )

        # ── Phase 8: Research pipeline stubs ──────────────────────────

        # Planner: decompose a research question into sub-tasks
        if "research planner" in prompt_lower and "sub-tasks" in prompt_lower:
            return (
                '["What is CNN?", "What is RNN?", "What is Transformer?", '
                '"Strengths and weaknesses comparison"]'
            )

        # Analyzer: transform evidence into structured findings
        if "research analyst" in prompt_lower and "structured findings" in prompt_lower:
            return (
                "- CNNs excel at spatial feature extraction using convolutional filters\n"
                "- RNNs process sequential data through recurrent connections\n"
                "- Transformers use self-attention for parallel sequence processing\n"
                "- Transformers have largely replaced RNNs for NLP tasks"
            )

        # Report writer: generate a structured research report
        if "research report writer" in prompt_lower and "executive summary" in prompt_lower:
            return (
                "Executive Summary\n"
                "This report compares CNN, RNN, and Transformer architectures.\n\n"
                "Detailed Findings\n"
                "CNNs excel at spatial feature extraction. "
                "RNNs process sequential data. "
                "Transformers use self-attention for parallel processing.\n\n"
                "Comparison\n"
                "Transformers outperform RNNs for most NLP tasks due to "
                "parallelization and long-range dependency handling.\n\n"
                "Conclusion\n"
                "Transformers are the current state-of-the-art for NLP.\n\n"
                "Recommendations\n"
                "Use Transformers for NLP tasks; consider CNNs for vision tasks."
            )

        # Partial report writer: when evidence is insufficient
        if "research report writer" in prompt_lower and "partial" in prompt_lower:
            return (
                "Partial Research Report\n"
                "Insufficient evidence was found to fully answer the question. "
                "The research process was unable to retrieve relevant documents."
            )

        return "Answer synthesized from the provided document context."


@lru_cache(maxsize=1)
def get_llm_service() -> LLMService:
    provider = os.getenv("LLM_PROVIDER", LLM_PROVIDER).lower()

    if provider == "groq":
        logger.info("Loading LLM provider: Groq / %s", GROQ_MODEL)
        return GroqLLMService(model=GROQ_MODEL, api_key=GROQ_API_KEY)

    if provider == "openai":
        logger.info("Loading LLM provider: OpenAI / %s", OPENAI_LLM_MODEL)
        return OpenAILLMService(model=OPENAI_LLM_MODEL, api_key=OPENAI_API_KEY)

    if provider == "test":
        logger.info("Loading LLM provider: TestLLMService (stub).")
        return TestLLMService()

    raise ValueError(
        f"Unsupported LLM_PROVIDER '{provider}'. "
        "Supported values: groq, openai, test."
    )


def generate_answer(prompt: str) -> str:
    """Generate an answer using the configured LLM provider."""
    return get_llm_service().generate(prompt)

"""Grounded RAG prompt construction."""

from app.schemas.retrieve import RetrieveResult

INSUFFICIENT_CONTEXT_MESSAGE = (
    "I could not find sufficient information in the provided documents."
)


def _format_context_blocks(chunks: list[RetrieveResult]) -> str:
    """Build numbered context blocks from retrieved chunks."""
    context_blocks: list[str] = []
    for index, chunk in enumerate(chunks, start=1):
        page_label = (
            f", page {chunk.page_number}" if chunk.page_number is not None else ""
        )
        context_blocks.append(
            f"[{index}] ({chunk.source_filename}{page_label}):\n{chunk.chunk_text}"
        )
    return "\n\n".join(context_blocks)


def build_rag_prompt(question: str, chunks: list[RetrieveResult]) -> str:
    """
    Build a grounded prompt that constrains the LLM to retrieved context only.

    Each chunk is labeled with its source filename and page number so the model
    can reason over traceable evidence. The prompt explicitly instructs the model
    to refuse when the context does not contain the answer.
    """
    context = _format_context_blocks(chunks)

    return (
        "You are a research assistant.\n\n"
        "Answer ONLY using the provided context.\n\n"
        "If the answer is not contained in the context, say:\n\n"
        f'"{INSUFFICIENT_CONTEXT_MESSAGE}"\n\n'
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer:"
    )


def build_conversational_rag_prompt(
    question: str,
    chunks: list[RetrieveResult],
    conversation_history: str,
) -> str:
    """
    Build a grounded RAG prompt that includes conversation history.

    The history lets the LLM resolve pronouns and references like "it",
    "they", "that technique", or "the previous paper" by providing the
    recent conversational context alongside the retrieved chunks.
    """
    context = _format_context_blocks(chunks)

    parts: list[str] = [
        "You are a research assistant.\n",
        "Answer ONLY using the provided context.\n",
        "Use the conversation history to understand references such as "
        '"it", "they", "this", or "that technique".\n',
        "If the answer is not contained in the context, say:\n",
        f'"{INSUFFICIENT_CONTEXT_MESSAGE}"\n',
    ]

    if conversation_history:
        parts.append(f"Conversation History:\n{conversation_history}\n")

    parts.append(f"Context:\n{context}\n")
    parts.append(f"Question: {question}\n")
    parts.append("Answer:")

    return "\n".join(parts)


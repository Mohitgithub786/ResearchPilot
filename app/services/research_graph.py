"""
LangGraph research workflow.

Defines a StateGraph with four processing nodes and two conditional edges
that orchestrate the agentic research pipeline:

    START → planner → retriever → check_retrieval → analyzer
                         ↑            ↓ (retry)       ↓
                         └────────────┘          check_findings
                         ↑                           ↓ (retry)
                         └───────────────────────────┘
                                                     ↓ (ok)
                                              report_writer → END

Retry logic
-----------
Both check_retrieval and check_findings can route back to the retriever
when their respective outputs are empty.  A shared retry_count (max 3)
prevents infinite loops.  After exhausting retries the graph proceeds to
the report_writer with whatever partial data is available.
"""

import json
from app.core.logging import get_logger

from langgraph.graph import END, StateGraph

from app.core.config import RESEARCH_RELEVANCE_THRESHOLD
from app.database.session import SessionLocal
from app.services.llm_service import generate_answer
from app.services.research_state import MAX_RETRIES, ResearchState
from app.services.retrieval_service import retrieve_chunks

INSUFFICIENT_CONTEXT_MESSAGE = (
    "Insufficient relevant information found in uploaded documents."
)

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Prompt templates for each node
# ---------------------------------------------------------------------------

PLANNER_PROMPT = """\
You are a research planner.

Given a complex research question, break it down into smaller, \
focused sub-tasks that can each be answered independently.

Return ONLY a JSON array of strings — no explanation, no markdown, \
no code fences.

Example:
["What is X?", "What is Y?", "Compare X and Y"]

Question: {query}

Sub-tasks:"""


ANALYZER_PROMPT = """\
You are a research analyst.

Analyze the following retrieved evidence and produce structured findings.

For each topic area covered in the evidence, write a concise finding \
that summarizes the key information.  Return each finding on its own \
line, prefixed with "- ".

Evidence:
{evidence}

Research sub-tasks being investigated:
{tasks}

Structured findings:"""


REPORT_WRITER_PROMPT = """\
You are a research report writer.

Using the findings below, write a professional research report that \
addresses the original question.

Structure your report with these sections:
1. Executive Summary
2. Detailed Findings
3. Comparison (if applicable)
4. Conclusion
5. Recommendations

Original question: {query}

Findings:
{findings}

Research Report:"""


PARTIAL_REPORT_PROMPT = """\
You are a research report writer.

The research process was unable to find sufficient evidence to fully \
answer the question.  Write a brief report acknowledging the limitations \
and presenting whatever partial information is available.

Original question: {query}

Available findings:
{findings}

Partial Research Report:"""


# ---------------------------------------------------------------------------
# Node implementations
# ---------------------------------------------------------------------------


def planner_node(state: ResearchState) -> dict:
    """
    Decompose the user's query into smaller research sub-tasks.

    Sends the query to the LLM with a decomposition prompt and parses the
    returned JSON list.  Falls back to treating the raw query as a single
    sub-task if JSON parsing fails.
    """
    query = state["query"]
    logger.info("PLANNER: decomposing query: %s", query)

    prompt = PLANNER_PROMPT.format(query=query)
    raw_response = generate_answer(prompt)
    logger.info("PLANNER: raw LLM response: %s", raw_response)

    try:
        plan = json.loads(raw_response)
        if not isinstance(plan, list):
            raise TypeError("Expected a list")
        plan = [str(task) for task in plan if str(task).strip()]
    except (json.JSONDecodeError, TypeError):
        logger.warning(
            "PLANNER: Could not parse JSON, using query as single task."
        )
        plan = [query]

    logger.info("PLANNER: generated %d sub-tasks: %s", len(plan), plan)
    return {"plan": plan, "status": "retrieving"}


def retriever_node(state: ResearchState) -> dict:
    """
    Retrieve evidence for every sub-task in the plan.

    Calls the existing retrieve_chunks() service for each sub-task, then
    aggregates and deduplicates the results.  Source citations are built
    from the chunk metadata.
    """
    plan = state["plan"]
    logger.info("RETRIEVER: retrieving for %d sub-tasks", len(plan))

    all_chunks: list[dict] = []
    all_sources: list[dict] = []
    seen_chunk_ids: set[int] = set()
    seen_source_keys: set[tuple[str, int | None]] = set()

    db = SessionLocal()
    try:
        for task in plan:
            logger.info("RETRIEVER: searching for: %s", task)
            results = retrieve_chunks(db, task)
            for result in results:
                if result.chunk_id not in seen_chunk_ids:
                    seen_chunk_ids.add(result.chunk_id)
                    all_chunks.append(
                        {
                            "chunk_id": result.chunk_id,
                            "document_id": result.document_id,
                            "page_number": result.page_number,
                            "source_filename": result.source_filename,
                            "similarity_score": result.similarity_score,
                            "chunk_text": result.chunk_text,
                        }
                    )
                source_key = (result.source_filename, result.page_number)
                if source_key not in seen_source_keys:
                    seen_source_keys.add(source_key)
                    all_sources.append(
                        {
                            "filename": result.source_filename,
                            "page_number": result.page_number,
                        }
                    )
    finally:
        db.close()

    avg_sim = 0.0
    if all_chunks:
        avg_sim = sum(c["similarity_score"] for c in all_chunks) / len(all_chunks)

    logger.info(
        "RETRIEVER: aggregated %d unique chunks, %d unique sources, avg_similarity=%.4f",
        len(all_chunks),
        len(all_sources),
        avg_sim,
    )
    return {
        "retrieved_chunks": all_chunks,
        "sources": all_sources,
        "average_similarity": avg_sim,
        "status": "analyzing",
    }


def analyzer_node(state: ResearchState) -> dict:
    """
    Transform raw retrieved chunks into structured research findings.

    Sends the aggregated chunk text and sub-task list to the LLM, which
    returns concise bullet-point findings for each topic area.
    """
    chunks = state["retrieved_chunks"]
    plan = state["plan"]
    logger.info(
        "ANALYZER: analyzing %d chunks across %d sub-tasks",
        len(chunks),
        len(plan),
    )

    evidence_parts: list[str] = []
    for i, chunk in enumerate(chunks, 1):
        source = chunk.get("source_filename", "unknown")
        page = chunk.get("page_number")
        page_label = f", page {page}" if page is not None else ""
        evidence_parts.append(
            f"[{i}] ({source}{page_label}):\n{chunk['chunk_text']}"
        )
    evidence_text = "\n\n".join(evidence_parts)
    tasks_text = "\n".join(f"- {task}" for task in plan)

    prompt = ANALYZER_PROMPT.format(evidence=evidence_text, tasks=tasks_text)
    raw_response = generate_answer(prompt)
    logger.info("ANALYZER: raw LLM response: %s", raw_response)

    # Parse bullet-point findings
    findings = [
        line.lstrip("- ").strip()
        for line in raw_response.strip().split("\n")
        if line.strip() and line.strip() != "-"
    ]

    logger.info("ANALYZER: produced %d findings", len(findings))
    return {"findings": findings, "status": "writing"}


def report_writer_node(state: ResearchState) -> dict:
    """
    Generate a professional research report from the structured findings.

    Uses the full report template when findings are available, or the
    partial report template when the pipeline exhausted its retries.
    """
    findings = state["findings"]
    query = state["query"]
    logger.info("REPORT WRITER: generating report from %d findings", len(findings))

    findings_text = "\n".join(f"- {f}" for f in findings) if findings else "(none)"

    if findings:
        prompt = REPORT_WRITER_PROMPT.format(
            query=query, findings=findings_text
        )
        status = "complete"
    else:
        prompt = PARTIAL_REPORT_PROMPT.format(
            query=query, findings=findings_text
        )
        status = "partial"

    report = generate_answer(prompt)
    logger.info("REPORT WRITER: report generated (%d chars), status=%s", len(report), status)
    return {"report": report, "status": status}


# ---------------------------------------------------------------------------
# Conditional edge functions
# ---------------------------------------------------------------------------


def check_retrieval(state: ResearchState) -> str:
    """
    Route after the retriever node.

    - no chunks + retries remain  → retry_retrieval
    - no chunks + max retries     → report_writer  (partial)
    - chunks but low similarity   → insufficient_context  (graceful exit)
    - chunks with good similarity → analyzer
    """
    if not state["retrieved_chunks"]:
        if state["retry_count"] < MAX_RETRIES:
            logger.warning(
                "CHECK_RETRIEVAL: no chunks, retry %d/%d → retriever",
                state["retry_count"] + 1,
                MAX_RETRIES,
            )
            return "retry_retrieval"
        logger.warning("CHECK_RETRIEVAL: max retries exhausted → report_writer")
        return "report_writer"

    avg_sim = state.get("average_similarity", 0.0)
    threshold = RESEARCH_RELEVANCE_THRESHOLD

    if avg_sim < threshold:
        logger.warning(
            "CHECK_RETRIEVAL: low relevance (avg_similarity=%.4f < threshold=%.2f) "
            "→ insufficient_context",
            avg_sim,
            threshold,
        )
        return "insufficient_context"

    logger.info(
        "CHECK_RETRIEVAL: good relevance (avg_similarity=%.4f >= threshold=%.2f) "
        "→ analyzer",
        avg_sim,
        threshold,
    )
    return "analyzer"


def check_findings(state: ResearchState) -> str:
    """
    Route after the analyzer node.

    - findings present       → report_writer
    - findings empty, retries → retriever  (re-retrieve)
    - findings empty, maxed   → report_writer  (partial report)
    """
    if state["findings"]:
        logger.info("CHECK_FINDINGS: findings present → report_writer")
        return "report_writer"

    if state["retry_count"] < MAX_RETRIES:
        logger.warning(
            "CHECK_FINDINGS: no findings, retry %d/%d → retriever",
            state["retry_count"] + 1,
            MAX_RETRIES,
        )
        return "retry_retrieval"

    logger.warning("CHECK_FINDINGS: max retries exhausted → report_writer")
    return "report_writer"


# ---------------------------------------------------------------------------
# Retry node — increments the counter before re-entering the retriever
# ---------------------------------------------------------------------------


def retry_retrieval_node(state: ResearchState) -> dict:
    """Increment the retry counter.  The graph routes here before looping."""
    new_count = state["retry_count"] + 1
    logger.info("RETRY: incrementing retry_count to %d", new_count)
    return {"retry_count": new_count, "status": "retrieving"}


def insufficient_context_node(state: ResearchState) -> dict:
    """
    Graceful exit when retrieved evidence is not relevant to the query.

    Sets a canned report, clears findings, and marks status as
    'insufficient_context' so the orchestrator returns a clean response
    without calling the analyzer or report writer.
    """
    logger.info(
        "INSUFFICIENT CONTEXT: avg_similarity=%.4f below threshold=%.2f, "
        "skipping analysis and report generation",
        state.get("average_similarity", 0.0),
        RESEARCH_RELEVANCE_THRESHOLD,
    )
    return {
        "report": INSUFFICIENT_CONTEXT_MESSAGE,
        "findings": [],
        "sources": [],
        "status": "insufficient_context",
    }


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------


def build_research_graph():
    """
    Build and compile the LangGraph research workflow.

    Graph topology:

        START → planner → retriever → (check_retrieval)
                              ↑            ↓
                      retry_retrieval  analyzer → (check_findings)
                              ↑                        ↓
                              └────────────────────────┘
                                                       ↓
                                                 report_writer → END
    """
    graph = StateGraph(ResearchState)

    # Register nodes
    graph.add_node("planner", planner_node)
    graph.add_node("retriever", retriever_node)
    graph.add_node("analyzer", analyzer_node)
    graph.add_node("report_writer", report_writer_node)
    graph.add_node("retry_retrieval", retry_retrieval_node)
    graph.add_node("insufficient_context", insufficient_context_node)

    # Entry point
    graph.set_entry_point("planner")

    # Unconditional edges
    graph.add_edge("planner", "retriever")
    graph.add_edge("retry_retrieval", "retriever")
    graph.add_edge("report_writer", END)
    graph.add_edge("insufficient_context", END)

    # Conditional edges
    graph.add_conditional_edges(
        "retriever",
        check_retrieval,
        {
            "analyzer": "analyzer",
            "retry_retrieval": "retry_retrieval",
            "report_writer": "report_writer",
            "insufficient_context": "insufficient_context",
        },
    )
    graph.add_conditional_edges(
        "analyzer",
        check_findings,
        {
            "report_writer": "report_writer",
            "retry_retrieval": "retry_retrieval",
        },
    )

    return graph.compile()

"""
LangGraph workflow for ECU agent.

This module defines the agent workflow with support for:
1. Single model queries
2. Comparison queries (concurrent retrieval)
3. Generic queries (no model constraint)
"""
import time
from typing import TypedDict, Callable, Optional, Dict, List
from langgraph.graph import StateGraph, END
from langchain_community.vectorstores import FAISS

from me_ecu_agent.agent.query_analysis import classify_query_intent, QueryIntent
from me_ecu_agent.agent.retrieval import (
    retrieve_for_models_sync,
    retrieve_generic_sync,
)
from me_ecu_agent.agent.util import (
    build_answer_prompt,
    build_compare_prompt,
    run_llm,
    format_context,
    format_contexts_for_compare,
)
from me_ecu_agent.query import query_chunks_by_model
from me_ecu_agent.query.config import QueryConfig
from me_ecu_agent.query.meta_store import MetaStore


# -----------------------------
# Agent state definition
# -----------------------------

class AgentState(TypedDict):
    """
    Shared state passed between graph nodes.
    """
    # Input
    query: str
    
    # Analysis
    intent: Optional[QueryIntent]
    
    # Retrieval
    context: str  # For single/generic queries
    contexts_by_model: Optional[Dict[str, str]]  # For compare queries
    
    # Output
    answer: str


# -----------------------------
# Timing decorator
# -----------------------------

def timed_node(fn: Callable, node_name: str, debug: bool = False) -> Callable:
    """
    Wrap a node function to measure and log execution time.
    """
    def wrapper(state):
        if debug:
            start = time.perf_counter()
            result = fn(state)
            elapsed_ms = (time.perf_counter() - start) * 1000
            print(f"[LangGraph][{node_name}] {elapsed_ms:.2f}ms")
            return result
    return wrapper


# -----------------------------
# Graph nodes
# -----------------------------

def analyze_query_node(state: AgentState) -> AgentState:
    """
    Analyze the query to determine intent and extract models.
    
    This is the first node in the workflow.
    """
    query = state["query"]
    intent = classify_query_intent(query)
    
    print(f"[Analysis] Intent: {intent.intent_type}, Models: {intent.models}")
    
    state["intent"] = intent
    return state


def retrieve_single_node(state: AgentState, *, vectorstore) -> AgentState:
    """
    Retrieve context for a single model query.
    
    Called when: intent.is_single_model == True
    """
    query = state["query"]
    intent = state["intent"]
    model = intent.models[0]
    
    print(f"[Retrieve Single] Model: {model}")
    
    # Retrieve chunks
    results = query_chunks_by_model(query, model, vectorstore, top_k=5)
    chunks = [doc.page_content for doc, _ in results]
    
    # Format context
    state["context"] = format_context(chunks)
    return state


def retrieve_compare_node(state: AgentState, *, vectorstore) -> AgentState:
    """
    Retrieve context for multiple models concurrently.
    
    Called when: intent.is_compare == True
    """
    query = state["query"]
    intent = state["intent"]
    models = intent.models
    
    print(f"[Retrieve Compare] Models: {models}")
    
    # Concurrent retrieval
    contexts_by_model = retrieve_for_models_sync(
        query, models, vectorstore, top_k=5
    )
    
    # Format contexts
    state["contexts_by_model"] = format_contexts_for_compare(
        contexts_by_model, max_chars_per_model=2000
    )
    
    return state


def retrieve_generic_node(state: AgentState, *, vectorstore) -> AgentState:
    """
    Retrieve context for generic queries (no model specified).
    
    Called when: intent.is_generic == True
    
    Note: This retrieves more chunks (top_k=10) for potential reranking.
    """
    query = state["query"]
    
    print(f"[Retrieve Generic]")
    
    # Retrieve chunks without model constraint
    chunks = retrieve_generic_sync(query, vectorstore, top_k=10)
    
    # TODO: Add cross-encoder reranking here
    # For now, just use top 5
    chunks = chunks[:5]
    
    # Format context
    state["context"] = format_context(chunks)
    return state


def answer_node(state: AgentState) -> AgentState:
    """
    Generate answer for single model or generic queries.
    
    Called when: intent.is_single_model or intent.is_generic
    """
    query = state["query"]
    context = state["context"]
    
    print(f"[Answer] Generating response...")
    
    prompt = build_answer_prompt(query, context)
    answer = run_llm(prompt, query=query, context=context)
    
    state["answer"] = answer
    return state


def compare_answer_node(state: AgentState) -> AgentState:
    """
    Generate comparative answer for comparison queries.
    
    Called when: intent.is_compare == True
    """
    query = state["query"]
    contexts = state["contexts_by_model"]
    
    print(f"[Compare Answer] Comparing {len(contexts)} models...")
    
    prompt = build_compare_prompt(query, contexts)
    answer = run_llm(prompt, query=query, context="\n\n".join(
        f"=== {model} ===\n{ctx}" for model, ctx in contexts.items()
    ))
    
    state["answer"] = answer
    return state


# -----------------------------
# Routing logic
# -----------------------------

def route_after_analysis(state: AgentState) -> str:
    """
    Route to the appropriate retrieval node based on intent.
    
    Returns:
        Node name to execute next.
    """
    intent = state["intent"]
    
    if intent.is_single_model:
        return "retrieve_single"
    elif intent.is_compare:
        return "retrieve_compare"
    else:  # generic
        return "retrieve_generic"


def route_after_retrieval(state: AgentState) -> str:
    """
    Route to the appropriate answer node based on intent.
    
    Returns:
        Node name to execute next.
    """
    intent = state["intent"]
    
    if intent.is_compare:
        return "compare_answer"
    else:
        return "answer"


# -----------------------------
# Graph construction
# -----------------------------

def build_graph(vectorstore: FAISS, query_config: QueryConfig, meta_store: MetaStore) -> StateGraph:
    """
    Build and compile the LangGraph workflow.
    
    Workflow:
        analyze_query →
            ├─ retrieve_single → answer → END
            ├─ retrieve_compare → compare_answer → END
            └─ retrieve_generic → answer → END
    
    Args:
        vectorstore: FAISS vector store instance.
    
    Returns:
        Compiled LangGraph application.
    """
    graph = StateGraph(AgentState)
    
    # -----------------------------
    # Register nodes with timing
    # -----------------------------
    
    graph.add_node(
        "analyze_query",
        timed_node(analyze_query_node, "analyze_query"),
    )
    
    graph.add_node(
        "retrieve_single",
        timed_node(
            lambda state: retrieve_single_node(state, vectorstore=vectorstore),
            "retrieve_single",
        ),
    )
    
    graph.add_node(
        "retrieve_compare",
        timed_node(
            lambda state: retrieve_compare_node(state, vectorstore=vectorstore),
            "retrieve_compare",
        ),
    )
    
    graph.add_node(
        "retrieve_generic",
        timed_node(
            lambda state: retrieve_generic_node(state, vectorstore=vectorstore),
            "retrieve_generic",
        ),
    )
    
    graph.add_node(
        "answer",
        timed_node(answer_node, "answer"),
    )
    
    graph.add_node(
        "compare_answer",
        timed_node(compare_answer_node, "compare_answer"),
    )
    
    # -----------------------------
    # Define edges with routing
    # -----------------------------
    
    # Entry point
    graph.set_entry_point("analyze_query")
    
    # Route after analysis
    graph.add_conditional_edges(
        "analyze_query",
        route_after_analysis,
        {
            "retrieve_single": "retrieve_single",
            "retrieve_compare": "retrieve_compare",
            "retrieve_generic": "retrieve_generic",
        },
    )
    
    # Single model path
    graph.add_edge("retrieve_single", "answer")
    
    # Compare path
    graph.add_edge("retrieve_compare", "compare_answer")
    
    # Generic path
    graph.add_edge("retrieve_generic", "answer")
    
    # All answer nodes lead to END
    graph.add_edge("answer", END)
    graph.add_edge("compare_answer", END)
    
    return graph.compile()
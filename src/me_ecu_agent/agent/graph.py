"""
LangGraph workflow for ECU agent.

This module defines the agent workflow with support for:
1. Single model queries
2. Comparison queries (concurrent retrieval)
3. Generic queries (no model constraint)
4. Spec-based cross-model comparison
"""
import time
import logging
from functools import wraps
from typing import TypedDict, Callable, Optional, Dict
from langgraph.graph import StateGraph, END

from me_ecu_agent.agent.query_analysis import classify_query_intent, QueryIntent
from me_ecu_agent.agent.agent_retrieval import (
    retrieve_for_models,
    retrieve_generic,
    retrieve_all_model_specs,
    retrieve_spec_for_model
)
from me_ecu_agent.agent.llm_util import (
    build_answer_prompt,
    build_compare_prompt,
    build_spec_comparison_prompt,
    run_llm,
    format_context,
    format_contexts_for_compare,
    format_specs_for_comparison,
)
from me_ecu_agent.query import Retriever

# Initialize logger
logger = logging.getLogger(__name__)


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
    specs_by_model: Optional[Dict[str, str]]  # For spec comparison queries
    
    # Output
    answer: str


# -----------------------------
# Timing decorator
# -----------------------------

def timed_node(func: Callable) -> Callable:
    """
    Decorator to measure and log node execution time.
    
    Usage:
        @timed_node
        def my_node(state: AgentState) -> AgentState:
            ...
    """
    @wraps(func)
    def wrapper(state):
        node_name = func.__name__.replace('_node', '').replace('_', ' ').title()
        
        logger.debug(f"[{node_name}] Starting...")
        start = time.perf_counter()
        
        result = func(state)
        
        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.debug(f"[{node_name}] Completed in {elapsed_ms:.2f}ms")
        
        return result
    
    return wrapper


# -----------------------------
# Graph nodes
# -----------------------------

@timed_node
def analyze_query_node(state: AgentState) -> AgentState:
    """
    Analyze the query to determine intent and extract models.
    
    This is the first node in the workflow.
    """
    query = state["query"]
    logger.debug(f"Analyzing query: {query}")
    
    intent = classify_query_intent(query)
    
    logger.info(
        f"Intent: {intent.intent_type}, Models: {intent.models}, "
        f"Requires Specs: {intent.requires_specs}"
    )
    
    state["intent"] = intent
    return state


def retrieve_single_node_impl(state: AgentState, retriever: Retriever, agent_config: dict) -> AgentState:
    """
    Retrieve context for a single model query.
    
    Called when: intent.is_single_model == True
    """
    query = state["query"]
    intent = state["intent"]
    model = intent.models[0]
    
    logger.info(f"Retrieving for single model: {model}")
    
    chunks = []
    
    # Retrieve spec chunks if needed
    if intent.requires_specs:
        logger.debug(f"Retrieving spec chunks for model {model}")
        spec_chunks = retrieve_spec_for_model(model, retriever, top_k=agent_config.get("spec_top_k", 2))
        chunks.extend(spec_chunks)
    
    # Always retrieve regular chunks
    logger.debug(f"Retrieving regular chunks for model {model}")
    model_results = retriever.query_by_model(model, query, top_k=agent_config.get("single_model_top_k", 3))
    model_chunks = [doc.page_content for doc, _ in model_results]
    chunks.extend(model_chunks)
    
    logger.debug(f"Retrieved {len(chunks)} total chunks for model {model}")
    
    # Format context
    state["context"] = format_context(chunks)
    return state


def retrieve_compare_node_impl(state: AgentState, retriever: Retriever, agent_config: dict) -> AgentState:
    """
    Retrieve context for multiple models.
    
    Called when: intent.is_compare == True
    """
    query = state["query"]
    intent = state["intent"]
    models = intent.models
    
    logger.info(f"Retrieving for comparison: {models}")
    
    # Retrieve for all models
    contexts_by_model = retrieve_for_models(
        query, models, retriever, top_k=agent_config.get("compare_top_k", 3)
    )
    
    logger.debug(f"Retrieved contexts for {len(contexts_by_model)} models")
    
    # Format contexts
    state["contexts_by_model"] = format_contexts_for_compare(
        contexts_by_model, max_chars_per_model=agent_config.get("max_chars_per_model", 2000)
    )
    
    return state


def retrieve_generic_node_impl(state: AgentState, retriever: Retriever, agent_config: dict) -> AgentState:
    """
    Retrieve context for generic queries (no model specified).
    
    Called when: intent.is_generic == True
    """
    query = state["query"]
    
    logger.info("Retrieving for generic query")
    
    # Retrieve chunks without model constraint
    chunks = retrieve_generic(query, retriever, top_k=agent_config.get("generic_top_k", 5))
    
    logger.debug(f"Retrieved {len(chunks)} chunks for generic query")
    
    # Format context
    state["context"] = format_context(chunks)
    return state


def retrieve_model_specs_node_impl(state: AgentState, retriever: Retriever, agent_config: dict) -> AgentState:
    """
    Retrieve specification chunks for ALL available models.
    
    Called when: intent.is_spec_comparison == True
    
    This is used for queries like:
    - "Which model has the highest temperature rating?"
    - "What is the most robust model?"
    - "Which ECU has the most RAM?"
    """
    query = state["query"]
    
    logger.info("Retrieving specification chunks for all models")
    
    # Retrieve spec chunks for all models
    specs_by_model = retrieve_all_model_specs(retriever, top_k=agent_config.get("spec_top_k", 2))
    
    logger.info(f"Retrieved specs for {len(specs_by_model)} models")
    
    # Format specs
    state["specs_by_model"] = format_specs_for_comparison(
        specs_by_model, max_chars_per_model=agent_config.get("max_chars_per_model_specs", 1000)
    )
    
    return state


@timed_node
def answer_node(state: AgentState) -> AgentState:
    """
    Generate answer for single model or generic queries.
    
    Called when: intent.is_single_model or intent.is_generic
    """
    query = state["query"]
    context = state["context"]
    
    logger.info("Generating answer...")
    
    prompt = build_answer_prompt(query, context)
    answer = run_llm(prompt, query=query, context=context)
        
    state["answer"] = answer
    return state


@timed_node
def compare_answer_node(state: AgentState) -> AgentState:
    """
    Generate comparative answer for comparison queries.
    
    Called when: intent.is_compare == True
    """
    query = state["query"]
    contexts = state["contexts_by_model"]
    
    logger.info(f"Generating comparison answer for {len(contexts)} models")
    
    prompt = build_compare_prompt(query, contexts)
    answer = run_llm(prompt, query=query, context="\n\n".join(
        f"=== {model} ===\n{ctx}" for model, ctx in contexts.items()
    ))
        
    state["answer"] = answer
    return state


@timed_node
def spec_comparison_answer_node(state: AgentState) -> AgentState:
    """
    Generate answer for spec-based cross-model comparison.
    
    Called when: intent.is_spec_comparison == True
    
    This analyzes specification tables to answer questions like:
    - "Which model has the highest operating temperature?"
    - "What's the most powerful ECU?"
    - "Which model is best for harsh environments?"
    """
    query = state["query"]
    specs = state["specs_by_model"]
    
    logger.info(f"Generating spec comparison answer for {len(specs)} models")
    
    prompt = build_spec_comparison_prompt(query, specs)
    answer = run_llm(prompt, query=query, specs="\n\n".join(
        f"=== {model} ===\n{spec}" for model, spec in specs.items()
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
        route = "retrieve_single"
    elif intent.is_compare:
        route = "retrieve_compare"
    elif intent.is_spec_comparison:
        route = "retrieve_model_specs"
    else:  # generic
        route = "retrieve_generic"
    
    logger.debug(f"Routing after analysis: {route}")
    return route


def route_after_retrieval(state: AgentState) -> str:
    """
    Route to the appropriate answer node based on intent.
    
    Returns:
        Node name to execute next.
    """
    intent = state["intent"]
    
    if intent.is_compare:
        route = "compare_answer"
    elif intent.is_spec_comparison:
        route = "spec_comparison_answer"
    else:
        route = "answer"
    
    logger.debug(f"Routing after retrieval: {route}")
    return route


# -----------------------------
# Graph construction
# -----------------------------

def build_graph(retriever: Retriever, agent_config: dict) -> StateGraph:
    """
    Build and compile the LangGraph workflow.
    
    Workflow:
        analyze_query →
            ├─ retrieve_single → answer → END
            ├─ retrieve_compare → compare_answer → END
            ├─ retrieve_generic → answer → END
            └─ retrieve_model_specs → spec_comparison_answer → END
    
    Args:
        retriever: Retriever instance from QueryFactory.
    
    Returns:
        Compiled LangGraph application.
    """
    logger.info("Building LangGraph workflow...")
    
    graph = StateGraph(AgentState)
    
    # -----------------------------
    # Register nodes
    # -----------------------------
    
    graph.add_node("analyze_query", analyze_query_node)
    
    # Retrieval nodes with timing wrapper
    @timed_node
    def retrieve_single_node(state):
        return retrieve_single_node_impl(state, retriever, agent_config)
    
    @timed_node
    def retrieve_compare_node(state):
        return retrieve_compare_node_impl(state, retriever, agent_config)
    
    @timed_node
    def retrieve_generic_node(state):
        return retrieve_generic_node_impl(state, retriever, agent_config)
    
    @timed_node
    def retrieve_model_specs_node(state):
        return retrieve_model_specs_node_impl(state, retriever, agent_config)
    
    graph.add_node("retrieve_single", retrieve_single_node)
    graph.add_node("retrieve_compare", retrieve_compare_node)
    graph.add_node("retrieve_generic", retrieve_generic_node)
    graph.add_node("retrieve_model_specs", retrieve_model_specs_node)
    
    # Answer nodes (already decorated)
    graph.add_node("answer", answer_node)
    graph.add_node("compare_answer", compare_answer_node)
    graph.add_node("spec_comparison_answer", spec_comparison_answer_node)
    
    # -----------------------------
    # Define edges with routing
    # -----------------------------
    
    # Entry point
    graph.set_entry_point("analyze_query")
    
    # Route after analysis (4-way routing)
    graph.add_conditional_edges(
        "analyze_query",
        route_after_analysis,
        {
            "retrieve_single": "retrieve_single",
            "retrieve_compare": "retrieve_compare",
            "retrieve_generic": "retrieve_generic",
            "retrieve_model_specs": "retrieve_model_specs",
        },
    )
    
    # Single model path
    graph.add_edge("retrieve_single", "answer")
    
    # Compare path
    graph.add_edge("retrieve_compare", "compare_answer")
    
    # Generic path
    graph.add_edge("retrieve_generic", "answer")
    
    # Spec comparison path
    graph.add_edge("retrieve_model_specs", "spec_comparison_answer")
    
    # All answer nodes lead to END
    graph.add_edge("answer", END)
    graph.add_edge("compare_answer", END)
    graph.add_edge("spec_comparison_answer", END)
    
    logger.info("LangGraph workflow built successfully")
    logger.debug(
        "Workflow paths: "
        "analyze → [retrieve_single|retrieve_compare|retrieve_generic|retrieve_model_specs] "
        "→ [answer|compare_answer|spec_comparison_answer] → END"
    )
    
    return graph.compile()
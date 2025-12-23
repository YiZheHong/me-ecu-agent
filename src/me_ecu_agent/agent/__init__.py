"""
Agent module for ECU question answering.

Provides a LangGraph-based agent with support for:
- Single model queries
- Comparison queries (concurrent retrieval)
- Generic technical queries
"""

from me_ecu_agent.agent.graph import build_graph

__all__ = [
    "build_graph",
]
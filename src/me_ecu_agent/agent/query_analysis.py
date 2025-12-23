"""
Query analysis: intent classification and model extraction.

This module analyzes user queries to determine:
1. What models are mentioned (if any)
2. What type of query it is (single, compare, generic)
"""
import re
from typing import List, Optional, Literal
from dataclasses import dataclass


# Model pattern: ECU-750, ECU-850, ECU-850b, etc.
MODEL_PATTERN = re.compile(r"\b(ECU)-(\d+)([a-zA-Z]*)\b", re.IGNORECASE)


@dataclass
class QueryIntent:
    """
    Structured representation of query intent.
    """
    intent_type: Literal["single_model", "compare", "generic"]
    models: List[str]  # Empty for generic queries
    
    @property
    def is_single_model(self) -> bool:
        return self.intent_type == "single_model"
    
    @property
    def is_compare(self) -> bool:
        return self.intent_type == "compare"
    
    @property
    def is_generic(self) -> bool:
        return self.intent_type == "generic"


def extract_models_from_query(query: str) -> List[str]:
    """
    Extract all ECU model identifiers from the query.
    
    Args:
        query: User query string.
    
    Returns:
        List of unique model identifiers in canonical form (e.g., ["ECU-750", "ECU-850b"]).
    
    Examples:
        >>> extract_models_from_query("What is the max temp for ECU-750?")
        ['ECU-750']
        
        >>> extract_models_from_query("Compare ECU-850 and ecu-850b")
        ['ECU-850', 'ECU-850b']
        
        >>> extract_models_from_query("What is CAN bus?")
        []
    """
    matches = MODEL_PATTERN.findall(query)
    
    # Canonicalize: ECU prefix uppercase, preserve suffix case
    models = [f"{prefix.upper()}-{digits}{suffix}" for prefix, digits, suffix in matches]
    
    # Deduplicate while preserving order
    seen = set()
    unique_models = []
    for model in models:
        if model not in seen:
            seen.add(model)
            unique_models.append(model)
    
    return unique_models


def is_comparison_query(query: str) -> bool:
    """
    Detect if the query is asking for a comparison.
    
    Uses heuristics based on common comparison keywords.
    
    Args:
        query: User query string.
    
    Returns:
        True if query appears to be a comparison.
    
    Examples:
        >>> is_comparison_query("Compare ECU-750 and ECU-850")
        True
        
        >>> is_comparison_query("What are the differences between ECU-850 and ECU-850b?")
        True
        
        >>> is_comparison_query("How much RAM does the ECU-850 have?")
        False
    """
    query_lower = query.lower()
    
    comparison_keywords = [
        "compare",
        "comparison",
        "difference",
        "differences",
        "vs",
        "versus",
        "between",
        "contrast",
        "which is better",
        "which one",
    ]
    
    return any(keyword in query_lower for keyword in comparison_keywords)


def classify_query_intent(query: str) -> QueryIntent:
    """
    Classify the query into one of three intent types.
    
    Classification logic:
    1. Extract all mentioned models
    2. If 0 models → generic
    3. If 1 model → single_model
    4. If 2+ models → compare (if comparison keywords present)
    5. If 2+ models but no comparison keywords → treat as single_model (use first)
    
    Args:
        query: User query string.
    
    Returns:
        QueryIntent object with intent_type and models.
    
    Examples:
        >>> classify_query_intent("What is the max temp for ECU-750?")
        QueryIntent(intent_type='single_model', models=['ECU-750'])
        
        >>> classify_query_intent("Compare ECU-750 and ECU-850")
        QueryIntent(intent_type='compare', models=['ECU-750', 'ECU-850'])
        
        >>> classify_query_intent("What is CAN bus?")
        QueryIntent(intent_type='generic', models=[])
    """
    models = extract_models_from_query(query)
    
    # Case 1: No models mentioned
    if len(models) == 0:
        return QueryIntent(intent_type="generic", models=[])
    
    # Case 2: Single model mentioned
    if len(models) == 1:
        return QueryIntent(intent_type="single_model", models=models)
    
    # Case 3: Multiple models + comparison keywords
    if is_comparison_query(query):
        return QueryIntent(intent_type="compare", models=models)
    
    # Case 4: Multiple models but no comparison intent
    # Treat as single_model query for the first mentioned model
    # (This is a fallback; most multi-model queries should have comparison keywords)
    return QueryIntent(intent_type="single_model", models=[models[0]])
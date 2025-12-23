"""
Query analysis: intent classification and model extraction.

This module analyzes user queries to determine:
1. What models are mentioned (if any)
2. What type of query it is (single, compare, generic, spec_comparison)
3. Whether the query requires specification data
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
    intent_type: Literal["single_model", "compare", "generic", "spec_comparison"]
    models: List[str]  # Empty for generic queries
    requires_specs: bool = False  # True if query needs specification data
    
    @property
    def is_single_model(self) -> bool:
        return self.intent_type == "single_model"
    
    @property
    def is_compare(self) -> bool:
        return self.intent_type == "compare"
    
    @property
    def is_generic(self) -> bool:
        return self.intent_type == "generic"
    
    @property
    def is_spec_comparison(self) -> bool:
        return self.intent_type == "spec_comparison"


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


def requires_spec_data(query: str) -> bool:
    """
    Detect if the query requires specification/technical data.
    
    This is used to determine if we need to retrieve spec chunks
    for cross-model comparison queries.
    
    Args:
        query: User query string.
    
    Returns:
        True if query needs specification data.
    
    Examples:
        >>> requires_spec_data("What is the most robust model in harsh environment?")
        True
        
        >>> requires_spec_data("Which model has the highest temperature rating?")
        True
        
        >>> requires_spec_data("Which model has more RAM?")
        True
        
        >>> requires_spec_data("What is CAN bus?")
        False
        
        >>> requires_spec_data("How do I configure the ECU-850?")
        False
    """
    query_lower = query.lower()
    
    # Technical specification keywords
    spec_keywords = [
        # Performance metrics
        "most robust",
        "highest",
        "lowest",
        "maximum",
        "minimum",
        "best performance",
        "fastest",
        "slowest",
        "more powerful",
        "less power",
        
        # Specific technical attributes
        "temperature",
        "temp",
        "operating temp",
        "ram",
        "memory",
        "storage",
        "processor",
        "cpu",
        "power consumption",
        "voltage",
        "current",
        "mbps",
        "bandwidth",
        "can fd",
        "can bus",
        "ethernet",
        "npu",
        "tops",
        
        # Comparative questions about specs
        "which model has",
        "which one has",
        "what model has",
        "does any model have",
        "which ecu has",
        
        # Superlative comparisons
        "better for",
        "suited for",
        "recommended for",
        "harsh environment",
        "high temperature",
        "low power",
        
        # Technical specifications
        "specification",
        "specs",
        "technical",
        "datasheet",
    ]
    
    return any(keyword in query_lower for keyword in spec_keywords)


def classify_query_intent(query: str) -> QueryIntent:
    """
    Classify the query into one of four intent types.
    
    Classification logic:
    1. Extract all mentioned models
    2. Check if query requires specs
    3. If 0 models + requires_specs → spec_comparison (cross-model spec query)
    4. If 0 models → generic
    5. If 1 model → single_model
    6. If 2+ models + comparison keywords → compare
    7. If 2+ models but no comparison keywords → treat as single_model (use first)
    
    Args:
        query: User query string.
    
    Returns:
        QueryIntent object with intent_type, models, and requires_specs flag.
    
    Examples:
        >>> classify_query_intent("What is the max temp for ECU-750?")
        QueryIntent(intent_type='single_model', models=['ECU-750'], requires_specs=True)
        
        >>> classify_query_intent("Compare ECU-750 and ECU-850")
        QueryIntent(intent_type='compare', models=['ECU-750', 'ECU-850'], requires_specs=False)
        
        >>> classify_query_intent("What is the most robust model?")
        QueryIntent(intent_type='spec_comparison', models=[], requires_specs=True)
        
        >>> classify_query_intent("What is CAN bus?")
        QueryIntent(intent_type='generic', models=[], requires_specs=False)
    """
    models = extract_models_from_query(query)
    needs_specs = requires_spec_data(query)
    
    # Case 1: No models mentioned + needs specs → spec comparison
    if len(models) == 0 and needs_specs:
        return QueryIntent(
            intent_type="spec_comparison",
            models=[],
            requires_specs=True
        )
    
    # Case 2: No models mentioned → generic
    if len(models) == 0:
        return QueryIntent(
            intent_type="generic",
            models=[],
            requires_specs=False
        )
    
    # Case 3: Single model mentioned
    if len(models) == 1:
        return QueryIntent(
            intent_type="single_model",
            models=models,
            requires_specs=needs_specs
        )
    
    # Case 4: Multiple models + comparison keywords
    if is_comparison_query(query):
        return QueryIntent(
            intent_type="compare",
            models=models,
            requires_specs=needs_specs
        )
    
    # Case 5: Multiple models but no comparison intent
    # Treat as single_model query for the first mentioned model
    return QueryIntent(
        intent_type="single_model",
        models=[models[0]],
        requires_specs=needs_specs
    )
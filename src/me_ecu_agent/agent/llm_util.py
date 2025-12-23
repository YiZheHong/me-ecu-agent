"""
Agent utilities: LLM management, prompt builders, and context formatting.
"""
import os
from typing import List, Dict
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv


# -----------------------------
# LLM initialization
# -----------------------------
load_dotenv()


def get_llm() -> ChatOpenAI:
    """
    Create and return the LLM instance used by the agent.
    
    Centralizes all model/API configuration.
    """
    return ChatOpenAI(
        model="deepseek-chat",
        openai_api_key=os.environ["DEEPSEEK_API_KEY"],
        openai_api_base=os.environ["DEEPSEEK_API_BASE"],
        temperature=0,
    )


# -----------------------------
# Prompt builders
# -----------------------------

def build_answer_prompt(query: str, context: str) -> ChatPromptTemplate:
    """
    Build prompt for single-model or generic queries.
    
    Strictly grounded in provided context.
    """
    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "You are a technical assistant answering questions about ECU products.\n"
                    "Answer the user's question using ONLY the provided context.\n"
                    "If the answer is not explicitly stated in the context, say:\n"
                    "\"The provided documents do not specify this information.\"\n\n"
                    "Be concise and precise. Use bullet points when listing specifications."
                ),
            ),
            (
                "human",
                (
                    "Question:\n"
                    "{query}\n\n"
                    "Context:\n"
                    "{context}\n\n"
                    "Answer:"
                ),
            ),
        ]
    )


def build_compare_prompt(query: str, contexts: Dict[str, str]) -> ChatPromptTemplate:
    """
    Build prompt for comparison queries.
    
    Args:
        query: User's comparison question.
        contexts: Dict mapping model_name -> retrieved_context.
    
    Example:
        contexts = {
            "ECU-750": "Context for ECU-750...",
            "ECU-850": "Context for ECU-850..."
        }
    """
    # Format contexts into a structured string
    context_blocks = []
    for model, context in contexts.items():
        context_blocks.append(f"=== {model} ===\n{context}")
    
    combined_context = "\n\n".join(context_blocks)
    
    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "You are a technical assistant comparing ECU products.\n"
                    "You have been provided with context for multiple models.\n\n"
                    "Your task:\n"
                    "1. Compare the models based on the user's question.\n"
                    "2. Use ONLY information from the provided contexts.\n"
                    "3. Organize your answer clearly, highlighting similarities and differences.\n"
                    "4. If information is missing for a model, explicitly state it.\n\n"
                    "Format your comparison using:\n"
                    "- Comparison tables when appropriate\n"
                    "- Bullet points for clarity\n"
                    "- Clear section headers"
                ),
            ),
            (
                "human",
                (
                    "Question:\n"
                    "{query}\n\n"
                    "Context for each model:\n"
                    "{context}\n\n"
                    "Please provide a detailed comparison:"
                ),
            ),
        ]
    )


def build_spec_comparison_prompt(query: str, specs: Dict[str, str]) -> ChatPromptTemplate:
    """
    Build prompt for spec-based cross-model comparison.
    
    This is used when the user asks comparative questions without specifying
    models, requiring analysis of technical specifications across all models.
    
    Args:
        query: User's question (e.g., "Which model has the highest temperature rating?")
        specs: Dict mapping model_name -> specification_context.
    
    Example:
        specs = {
            "ECU-750": "| Feature | Specification |...",
            "ECU-850": "| Feature | Specification |...",
            "ECU-850b": "| Feature | Specification |..."
        }
    """
    # Format specs into a structured string
    spec_blocks = []
    for model, spec in specs.items():
        spec_blocks.append(f"=== {model} Technical Specifications ===\n{spec}")
    
    combined_specs = "\n\n".join(spec_blocks)
    
    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "You are a technical assistant analyzing ECU product specifications.\n"
                    "You have been provided with technical specification tables for all available models.\n\n"
                    "Your task:\n"
                    "1. Analyze the specifications to answer the user's question.\n"
                    "2. Use ONLY the information in the provided specification tables.\n"
                    "3. Provide concrete data from the specs to support your answer.\n"
                    "4. If the requested information is not in the specs, clearly state that.\n\n"
                    "Guidelines:\n"
                    "- Quote exact values from the specification tables\n"
                    "- Create comparison tables when comparing multiple attributes\n"
                    "- Highlight the key differences that answer the question\n"
                    "- Recommend specific models when appropriate based on the specs\n"
                    "- Be precise with units and ranges"
                ),
            ),
            (
                "human",
                (
                    "Question:\n"
                    "{query}\n\n"
                    "Technical Specifications:\n"
                    "{specs}\n\n"
                    "Please analyze the specifications and answer the question:"
                ),
            ),
        ]
    )


# -----------------------------
# LLM execution
# -----------------------------

def run_llm(prompt: ChatPromptTemplate, **kwargs) -> str:
    """
    Execute the LLM with the given prompt and inputs.
    
    Args:
        prompt: A ChatPromptTemplate.
        **kwargs: Variables to format into the prompt.
    
    Returns:
        The model's generated answer as plain text.
    """
    llm = get_llm()
    messages = prompt.format_messages(**kwargs)
    response = llm.invoke(messages)
    return response.content.strip()


# -----------------------------
# Context formatting
# -----------------------------

def format_context(chunks: List[str], max_chars: int = 4000) -> str:
    """
    Format retrieved chunks into a single context string.
    
    Args:
        chunks: List of chunk texts.
        max_chars: Maximum total characters allowed.
    
    Returns:
        Formatted context string with chunk boundaries preserved.
    """
    parts = []
    total = 0
    
    for i, chunk in enumerate(chunks, start=1):
        block = f"[Chunk {i}]\n{chunk.strip()}\n"
        if total + len(block) > max_chars:
            break
        parts.append(block)
        total += len(block)
    
    return "\n".join(parts)


def format_contexts_for_compare(
    contexts_by_model: Dict[str, List[str]],
    max_chars_per_model: int = 2000,
) -> Dict[str, str]:
    """
    Format contexts for comparison queries.
    
    Args:
        contexts_by_model: Dict mapping model -> list of chunks.
        max_chars_per_model: Max chars per model's context.
    
    Returns:
        Dict mapping model -> formatted context string.
    """
    formatted = {}
    
    for model, chunks in contexts_by_model.items():
        formatted[model] = format_context(chunks, max_chars=max_chars_per_model)
    
    return formatted


def format_specs_for_comparison(
    specs_by_model: Dict[str, List[str]],
    max_chars_per_model: int = 3000,
) -> Dict[str, str]:
    """
    Format specification chunks for cross-model comparison.
    
    Specs are given more space than regular chunks since they contain
    dense tabular information.
    
    Args:
        specs_by_model: Dict mapping model -> list of spec chunks.
        max_chars_per_model: Max chars per model's specs.
    
    Returns:
        Dict mapping model -> formatted specification string.
    """
    formatted = {}
    
    for model, chunks in specs_by_model.items():
        # For specs, we want to preserve the full table structure
        # so we're less aggressive about truncation
        formatted[model] = format_context(chunks, max_chars=max_chars_per_model)
    
    return formatted
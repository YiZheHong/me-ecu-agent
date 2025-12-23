"""
Agent utilities: LLM management, prompt builders, and context formatting.
"""
import os
from typing import List, Dict, Any
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
    
    Strictly grounded in provided context with source attribution.
    """
    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "You are a technical assistant answering questions about ECU products.\n"
                    "Provide CONCISE, factual answers using ONLY the provided context.\n\n"
                    "RESPONSE FORMAT (MANDATORY):\n"
                    "1. Direct answer in 1-2 sentences (use **bold** for key values)\n"
                    "2. End with: **Source:** [Source: filename | Section: section]\n\n"
                    "STRICT RULES:\n"
                    "- NO introductory phrases or preamble\n"
                    "- NO phrases like 'According to...', 'Based on...'\n"
                    "- Start directly with the answer\n"
                    "- Keep total response under 8 lines\n"
                    "- Use exact values and ranges from context\n"
                    "- When the question is related to specifications, list whole information mentioned in context, such as range or idle.\n"
                    "- If not in context, say: \"Not specified in provided documents.\""
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
                    "Provide CONCISE, well-structured comparisons using ONLY the provided context.\n\n"
                    "RESPONSE FORMAT (MANDATORY):\n"
                    "1. Start with a ONE-SENTENCE summary of the key difference\n"
                    "2. TWO-SENTENCE explanation of your comparison\n"
                    "3. End with source citations (one line per source)\n\n"
                    "STRICT RULES:\n"
                    "- NO introductory phrases like 'The ECU-850 provides...'\n"
                    "- NO repetition of the same information\n"
                    "- NO lengthy explanations - just facts and values\n"
                    "- Keep the total response under 10 lines\n"
                    "- Use exact values from specifications\n"
                    "- Always cite: [Source: filename | Section: section_name]"
                ),
            ),
            (
                "human",
                (
                    "Question:\n"
                    "{query}\n\n"
                    "Context for each model:\n"
                    "{context}\n\n"
                    "Provide a concise comparison:"
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
                    "You are a technical assistant analyzing ECU specifications.\n"
                    "Provide CONCISE spec comparisons using ONLY the provided context.\n\n"
                    "RESPONSE FORMAT (MANDATORY):\n"
                    "1. ONE-SENTENCE direct answer to the question\n"
                    "2. TWO-SENTENCE explanation of your reasoning based on specs\n"
                    "3. Source citations (one line)\n\n"
                    "STRICT RULES:\n"
                    "- NO introductory text or explanations\n"
                    "- Start directly with the answer\n"
                    "- Quote EXACT values with units from spec context\n"
                    "- Keep response under 10 lines total\n"
                    "- If spec not found, state: \"Specification not available\"\n"
                    "- Always cite: [Source: filename | Section: section]"
                ),
            ),
            (
                "human",
                (
                    "Question:\n"
                    "{query}\n\n"
                    "Technical Specifications:\n"
                    "{specs}\n\n"
                    "Analyze and answer:"
                ),
            ),
        ]
    )

def build_eval_prompt(expected_answer: str, actual_answer: str, evaluation_criteria: str) -> ChatPromptTemplate:
    """
    Build prompt for evaluating the LLM's answer.
    """
    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "You are an expert evaluator assessing the quality of answers provided by a technical assistant.\n"
                    "Your task is to determine if the assistant's answer meets the expected answer based on specific evaluation criteria.\n\n"
                    "Guidelines:\n"
                    "- Compare the actual answer against the expected answer.\n"
                    "- Use the evaluation criteria to guide your assessment.\n"
                    "- Provide a clear PASS/FAIL\n"
                    "- If all the key information are mentioned and correct in the actual answer, give a PASS.\n"
                ),
            ),
            (
                "human",
                (
                    "Expected Answer:\n"
                    "{expected_answer}\n\n"
                    "Actual Answer:\n"
                    "{actual_answer}\n\n"
                    "Evaluation Criteria:\n"
                    "{evaluation_criteria}\n\n"
                    "Based on the above, does the actual answer meet the expected answer? Give only a single word: PASS or FAIL"
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

def format_context(chunks: List[Dict[str, Any]], max_chars: int = 4000) -> str:
    """
    Format retrieved chunks into a single context string with metadata.
    
    Args:
        chunks: List of chunk dicts with 'content', 'source_filename', 
                'section_title', and 'status' keys.
        max_chars: Maximum total characters allowed.
    
    Returns:
        Formatted context string with source attribution for each chunk.
    """
    parts = []
    total = 0
    
    for i, chunk in enumerate(chunks, start=1):
        # Format chunk with metadata
        source_info = (
            f"[Source: {chunk['source_filename']} | "
            f"Section: {chunk['section_title']} | "
            f"Status: {chunk['status']}]"
        )
        block = f"[Chunk {i}]\n{source_info}\n{chunk['content'].strip()}\n"
        
        if total + len(block) > max_chars:
            break
        parts.append(block)
        total += len(block)
    
    return "\n".join(parts)


def format_contexts_for_compare(
    contexts_by_model: Dict[str, List[Dict[str, Any]]],
    max_chars_per_model: int = 2000,
) -> Dict[str, str]:
    """
    Format contexts for comparison queries with metadata.
    
    Args:
        contexts_by_model: Dict mapping model -> list of chunk dicts.
        max_chars_per_model: Max chars per model's context.
    
    Returns:
        Dict mapping model -> formatted context string.
    """
    formatted = {}
    
    for model, chunks in contexts_by_model.items():
        formatted[model] = format_context(chunks, max_chars=max_chars_per_model)
    
    return formatted


def format_specs_for_comparison(
    specs_by_model: Dict[str, List[Dict[str, Any]]],
    max_chars_per_model: int = 3000,
) -> Dict[str, str]:
    """
    Format specification chunks for cross-model comparison with metadata.
    
    Specs are given more space than regular chunks since they contain
    dense tabular information.
    
    Args:
        specs_by_model: Dict mapping model -> list of spec chunk dicts.
        max_chars_per_model: Max chars per model's specs.
    
    Returns:
        Dict mapping model -> formatted specification string.
    """
    formatted = {}
    
    for model, chunks in specs_by_model.items():
        # For specs, we want to preserve the full structure
        # so we're less aggressive about truncation
        formatted[model] = format_context(chunks, max_chars=max_chars_per_model)
    
    return formatted
# src/me_ecu_agent/agent/graph.py

import os
from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

from me_ecu_agent.rag.mini_rag import build_vectorstore

load_dotenv()

# -------- State --------
class AgentState(TypedDict):
    query: str
    targets: List[str]
    context: str
    answer: str

# -------- Model --------
llm = ChatOpenAI(
    model="deepseek-chat",
    openai_api_key=os.environ["DEEPSEEK_API_KEY"],
    openai_api_base=os.environ["DEEPSEEK_API_BASE"],
    temperature=0,
)

# -------- Router --------
def router(state: AgentState) -> AgentState:
    q = state["query"]
    targets = []

    if "750" in q or "700" in q:
        targets.append("ECU-750")
    if "800a" in q:
        targets.append("ECU-800a")
    if "800b" in q:
        targets.append("ECU-800b")
    if not targets:
        targets = ["ALL"]

    state["targets"] = targets
    return state


# -------- Retrieve + Answer --------
def retrieve_and_answer(state: AgentState) -> AgentState:
    docs = load_docs()
    vs = build_vectorstore(docs)

    query = state["query"]
    targets = state["targets"]

    results = vs.similarity_search(query, k=5)

    filtered = []

    for d in results:
        if "ALL" in targets:
            filtered.append(d)
        elif d.metadata.get("model") in targets:
            filtered.append(d)

    context = "\n\n".join(d.page_content for d in filtered)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Answer only using the provided context. "
                "If the answer is not in the context, say 'Not specified in documentation.'",
            ),
            ("human", "Context:\n{context}\n\nQuestion:\n{question}"),
        ]
    )

    answer = llm.invoke(
        prompt.format(context=context, question=query)
    ).content

    state["context"] = context
    state["answer"] = answer
    return state


# -------- Build Graph --------
def build_graph():
    g = StateGraph(AgentState)
    g.add_node("router", router)
    g.add_node("rag", retrieve_and_answer)

    g.set_entry_point("router")
    g.add_edge("router", "rag")
    g.add_edge("rag", END)

    return g.compile()

from me_ecu_agent.ingest.ingest import ingest
from me_ecu_agent.agent.graph import build_graph

vectorstore = ingest(rebuild=True)
# app = build_graph(vectorstore)

# result = app.invoke(
#     {
#         "query": "How do you enable the NPU on the ECU-850b?",
#         "context": "",
#         "answer": "",
#     }
# )

# print(result["answer"])

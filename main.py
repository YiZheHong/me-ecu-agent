from me_ecu_agent.ingest.ingest import ingest
from me_ecu_agent.query.query import query_chunks

vec = ingest(rebuild = True)
# print(len(vec.index_to_docstore_id))
# docs = query_chunks("What is the power consumption of the ECU-850b under load?", vec)

# for d in docs:
#     print(d.page_content)
#     print("-----")

# graph = build_graph()

# out = graph.invoke(
#     {"query": "Compare the CAN bus speed of the ECU-750 and the ECU-850"}
# )

# print(out["answer"])

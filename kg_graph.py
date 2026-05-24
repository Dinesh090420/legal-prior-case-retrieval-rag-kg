import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd

# 👉 your file path
path = "/Users/garbhapudinesh/Desktop/dl_project 3/output/kg_v2.graphml"

# load graph
G = nx.read_graphml(path)

# =========================
# 🔥 1. SAVE GRAPH IMAGE
# =========================
plt.figure(figsize=(10, 8))
pos = nx.spring_layout(G, k=0.15)

nx.draw(G, pos, node_size=10, with_labels=False)
plt.savefig("/Users/garbhapudinesh/Desktop/dl_project 3/output/kg_visual.png", dpi=300)
plt.close()

# =========================
# 🔥 2. SAVE NODES CSV
# =========================
nodes = []
for n, d in G.nodes(data=True):
    row = {"id": n}
    row.update(d)
    nodes.append(row)

pd.DataFrame(nodes).to_csv(
    "/Users/garbhapudinesh/Desktop/dl_project 3/output/nodes.csv",
    index=False
)

# =========================
# 🔥 3. SAVE EDGES CSV
# =========================
edges = []
for u, v, d in G.edges(data=True):
    row = {"source": u, "target": v}
    row.update(d)
    edges.append(row)

pd.DataFrame(edges).to_csv(
    "/Users/garbhapudinesh/Desktop/dl_project 3/output/edges.csv",
    index=False
)

print("✅ Done bro → files created:")
print("kg_visual.png")
print("nodes.csv")
print("edges.csv")
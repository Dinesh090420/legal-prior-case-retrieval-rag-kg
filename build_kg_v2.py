"""
Knowledge Graph v2 for IL-TUR PCR — train split only, NO ground-truth edges.

Changes from v1 (build_knowledge_graph.py):
  - Loads ONLY train_queries + train_candidates (eliminates data leakage)
  - NO CITES_PRECEDENT edges (ground-truth labels never enter the graph)
  - Keeps entity-linking edges: APPLIES_STATUTE, PRESIDED_BY, HEARD_IN, REFERENCES_CASE
  - Keeps derived co-occurrence edges: SHARES_STATUTE, SHARES_JUDGE
  - Processes full document text, not just first 5000 chars

Output: output/kg_v2.graphml
"""

import re
import json
import logging
from pathlib import Path
from collections import defaultdict

import pandas as pd
import networkx as nx

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

DATA_DIR = Path(".")
OUTPUT_DIR = Path("output")

STATUTE_PATTERN = re.compile(
    r"(?:Section|Sections|S\.)\s+"
    r"(\d+[A-Z]?(?:\s*[\(/]\s*\d+\s*[\)/])?)"
    r"(?:\s+(?:of|of the)\s+)?"
    r"((?:Indian Penal Code|IPC|Code of Criminal Procedure|Cr\.?\s*P\.?\s*C\.?|"
    r"Code of Civil Procedure|C\.?\s*P\.?\s*C\.?|Constitution|"
    r"Evidence Act|Contract Act|Companies Act|"
    r"Arms Act|NDPS Act|Motor Vehicles Act|"
    r"(?:the\s+)?[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+Act(?:,?\s*\d{4})?))?",
    re.IGNORECASE,
)

JUDGE_PATTERN = re.compile(
    r"(?:(?:Hon'?(?:ou)?ble|Mr\.?|Mrs\.?|Ms\.?|Justice|Dr\.?)\s+)+"
    r"([A-Z][a-z]+(?:\s+[A-Z]\.?)*(?:\s+[A-Z][a-z]+)+)",
)

COURT_PATTERN = re.compile(
    r"(Supreme Court of India|"
    r"High Court of [\w\s]+?(?=\s+(?:in|at|dated|\.|,))|"
    r"(?:Bombay|Delhi|Madras|Calcutta|Allahabad|Karnataka|"
    r"Kerala|Gujarat|Rajasthan|Punjab|Patna|Orissa|"
    r"Andhra Pradesh|Telangana|Madhya Pradesh|Chhattisgarh|"
    r"Jharkhand|Uttarakhand|Himachal Pradesh|Jammu|"
    r"Gauhati|Sikkim|Manipur|Meghalaya|Tripura)\s+High Court|"
    r"District Court|Sessions Court|Trial Court|"
    r"Tribunal|Commission)",
    re.IGNORECASE,
)

CASE_NO_PATTERN = re.compile(
    r"(?:Civil|Criminal|Writ|SLP|Special Leave)?\s*"
    r"(?:Appeal|Petition|Application|Case|Suit|Complaint)?\s*"
    r"(?:\(?\w+\)?\s+)?"
    r"No\.?\s*(\d+(?:\s*/\s*\d+)?)\s+(?:of\s+)?(\d{4})",
    re.IGNORECASE,
)


def _doc_text(row) -> str:
    t = row["text"]
    return t if isinstance(t, str) else " ".join(str(s) for s in t)


def load_train_documents() -> pd.DataFrame:
    """Load only train queries + train candidates — no dev/test leakage."""
    frames = []
    for name, role in [("train_queries", "query"), ("train_candidates", "candidate")]:
        path = DATA_DIR / f"{name}-00000-of-00001.parquet"
        if not path.exists():
            log.warning("Missing %s", path)
            continue
        df = pd.read_parquet(path)
        df["role"] = role
        frames.append(df)
        log.info("Loaded %s: %d docs", name, len(df))
    return pd.concat(frames, ignore_index=True)


def build_graph(docs: pd.DataFrame) -> nx.Graph:
    G = nx.Graph()
    entity_to_cases: dict[str, list[str]] = defaultdict(list)
    total = len(docs)

    for idx, row in docs.iterrows():
        if idx % 500 == 0:
            log.info("Processing document %d / %d", idx, total)

        case_id = str(row["id"])
        full_text = _doc_text(row)

        G.add_node(case_id, type="Case", role=row["role"])

        for m in STATUTE_PATTERN.finditer(full_text):
            section = m.group(1).strip()
            act = re.sub(r"\s+", " ", (m.group(2) or "").strip())
            label = f"S.{section}" + (f" {act}" if act else "")
            node_id = f"statute:{label}"
            if not G.has_node(node_id):
                G.add_node(node_id, type="Statute", section=section, act=act, label=label)
            G.add_edge(case_id, node_id, type="APPLIES_STATUTE")
            entity_to_cases[node_id].append(case_id)

        for m in JUDGE_PATTERN.finditer(full_text):
            name = m.group(1).strip()
            if len(name) <= 4 or any(w in name.lower() for w in ["court", "india", "bench"]):
                continue
            node_id = f"judge:{name}"
            if not G.has_node(node_id):
                G.add_node(node_id, type="Judge", name=name)
            G.add_edge(case_id, node_id, type="PRESIDED_BY")
            entity_to_cases[node_id].append(case_id)

        for m in COURT_PATTERN.finditer(full_text):
            court = re.sub(r"\s+", " ", m.group(1).strip())
            node_id = f"court:{court}"
            if not G.has_node(node_id):
                G.add_node(node_id, type="Court", name=court)
            G.add_edge(case_id, node_id, type="HEARD_IN")

        for m in CASE_NO_PATTERN.finditer(full_text):
            cn = f"{m.group(1).replace(' ', '')}/{m.group(2)}"
            node_id = f"caseno:{cn}"
            if not G.has_node(node_id):
                G.add_node(node_id, type="CaseNumber", number=cn)
            G.add_edge(case_id, node_id, type="REFERENCES_CASE")

    log.info("Building co-occurrence edges (SHARES_STATUTE, SHARES_JUDGE)...")
    for entity_id, case_ids in entity_to_cases.items():
        if len(case_ids) < 2 or len(case_ids) > 200:
            continue
        etype = G.nodes[entity_id].get("type", "")
        edge_label = "SHARES_STATUTE" if etype == "Statute" else "SHARES_JUDGE"
        for i in range(len(case_ids)):
            for j in range(i + 1, len(case_ids)):
                a, b = case_ids[i], case_ids[j]
                if G.has_edge(a, b):
                    G[a][b]["shared_count"] = G[a][b].get("shared_count", 1) + 1
                    G[a][b]["type"] = G[a][b].get("type", edge_label) + "," + edge_label
                else:
                    G.add_edge(a, b, type=edge_label, shared_entity=entity_id, shared_count=1)

    return G


def main():
    log.info("=== KG v2: Train-only, NO ground-truth edges ===")

    docs = load_train_documents()
    log.info("Total train documents: %d", len(docs))

    G = build_graph(docs)

    node_types = defaultdict(int)
    for _, d in G.nodes(data=True):
        node_types[d.get("type", "Unknown")] += 1
    edge_types = defaultdict(int)
    for _, _, d in G.edges(data=True):
        for t in d.get("type", "Unknown").split(","):
            edge_types[t.strip()] += 1

    stats = {
        "total_nodes": G.number_of_nodes(),
        "total_edges": G.number_of_edges(),
        "node_types": dict(node_types),
        "edge_types": dict(edge_types),
        "data_split": "train_only",
        "ground_truth_edges": False,
    }

    OUTPUT_DIR.mkdir(exist_ok=True)
    nx.write_graphml(G, str(OUTPUT_DIR / "kg_v2.graphml"))
    with open(OUTPUT_DIR / "kg_v2_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    log.info("=== KG v2 Summary ===")
    for k, v in stats.items():
        log.info("  %s: %s", k, v)
    log.info("Exported to output/kg_v2.graphml")


if __name__ == "__main__":
    main()

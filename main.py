"""
RA-KG-LLM v2: Memory-optimized + RAG-only JSON output
======================================================
- All 5 models run and are evaluated for metrics/plots
- JSON ranking files saved ONLY for RAG-BM25 and RAG+LLM:
      output/rankings/rag_bm25_rankings.json
      output/rankings/rag_llm_rankings.json
      output/rankings/per_query_rag_models_top5_<split>.json   ← TOP-5 ALL QUERIES
- Terminal print shows RAG model rankings for ALL queries (top 5)
"""

import argparse
import json
import math
import logging
import os
import re
import time
from collections import defaultdict
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

DATA_DIR   = Path(__file__).parent / "data"
OUTPUT_DIR = Path("output")
PLOTS_DIR  = OUTPUT_DIR / "plots"
RANK_DIR   = OUTPUT_DIR / "rankings"

# Only these models will be written to JSON files
RAG_MODELS = ["RAG-BM25", "RAG+LLM"]

# ===================================================================
# BM25 helper
# ===================================================================

class BM25VectorizerIndex:
    def __init__(self, docs, max_features=40000, k1=1.5, b=0.75):
        self.k1 = k1
        self.b  = b
        self.vectorizer = TfidfVectorizer(
            max_features=max_features, sublinear_tf=False, ngram_range=(1, 2)
        )
        tf_matrix      = self.vectorizer.fit_transform(docs)
        self.idf       = self.vectorizer.idf_
        dl             = tf_matrix.sum(axis=1).A1
        self.avgdl     = dl.mean() if dl.mean() > 0 else 1.0
        tf_arr         = tf_matrix.toarray().astype(np.float32)
        tf_sat         = (tf_arr * (k1 + 1)) / (
            tf_arr + k1 * (1 - b + b * (dl[:, None] / self.avgdl))
        )
        self.bm25_matrix = tf_sat * self.idf[None, :]
        normed           = normalize(self.bm25_matrix)
        self.index       = faiss.IndexFlatIP(normed.shape[1])
        self.index.add(normed)
        log.info("  BM25 index: %d docs, dim=%d, avgdl=%.0f",
                 self.index.ntotal, normed.shape[1], self.avgdl)

    def encode_query(self, text):
        tf_q   = self.vectorizer.transform([text]).toarray().astype(np.float32)[0]
        tf_sat = (tf_q * (self.k1 + 1)) / (tf_q + self.k1 * (1 - self.b + self.b))
        return normalize((tf_sat * self.idf)[None, :]).astype(np.float32)

    def search(self, query_text, top_k):
        qvec = self.encode_query(query_text)
        scores, indices = self.index.search(qvec, min(top_k, self.index.ntotal))
        return scores[0], indices[0]


# ===================================================================
# IR Evaluation Metrics
# ===================================================================

def precision_at_k(retrieved, relevant, k):
    top = retrieved[:k]
    return len(set(top) & relevant) / len(top) if top else 0.0

def recall_at_k(retrieved, relevant, k):
    return len(set(retrieved[:k]) & relevant) / len(relevant) if relevant else 0.0

def f1_at_k(retrieved, relevant, k):
    p, r = precision_at_k(retrieved, relevant, k), recall_at_k(retrieved, relevant, k)
    return 2 * p * r / (p + r) if (p + r) else 0.0

def average_precision(retrieved, relevant):
    if not relevant:
        return 0.0
    hits, total = 0, 0.0
    for i, doc in enumerate(retrieved):
        if doc in relevant:
            hits += 1
            total += hits / (i + 1)
    return total / len(relevant)

def ndcg_at_k(retrieved, relevant, k):
    top  = retrieved[:k]
    dcg  = sum(1.0 / math.log2(i + 2) for i, d in enumerate(top) if d in relevant)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / idcg if idcg else 0.0

def hits_at_k(retrieved, relevant, k):
    return 1.0 if set(retrieved[:k]) & relevant else 0.0


# ===================================================================
# evaluate_from_rankings
# ===================================================================

def evaluate_from_rankings(rankings: dict, queries: list, k_values: list) -> dict:
    metrics     = {k: defaultdict(list) for k in k_values}
    ap_list, h1 = [], []
    for q in queries:
        qid    = str(q["id"])
        rel    = set(str(c) for c in q["relevant_candidates"])
        ranked = rankings.get(qid, [])
        ap_list.append(average_precision(ranked, rel))
        h1.append(hits_at_k(ranked, rel, 1))
        for k in k_values:
            metrics[k]["P"].append(precision_at_k(ranked, rel, k))
            metrics[k]["R"].append(recall_at_k(ranked, rel, k))
            metrics[k]["F1"].append(f1_at_k(ranked, rel, k))
            metrics[k]["nDCG"].append(ndcg_at_k(ranked, rel, k))
            metrics[k]["Hits"].append(hits_at_k(ranked, rel, k))
    out = {}
    for k in k_values:
        for m in ("P", "R", "F1", "nDCG", "Hits"):
            out[f"{m}@{k}"] = float(np.mean(metrics[k][m]))
    out["MAP"]    = float(np.mean(ap_list))
    out["Hits@1"] = float(np.mean(h1))
    return out


# ===================================================================
# run_and_collect
# ===================================================================

def run_and_collect(retrieval_fn, queries: list, candidate_ids: list,
                    label: str = "") -> dict:
    log.info("  Collecting rankings for %s ...", label)
    rankings = {}
    t0       = time.time()
    for i, q in enumerate(queries):
        qid     = str(q["id"])
        ranked  = retrieval_fn(qid, candidate_ids)
        seen    = set(ranked)
        missing = [c for c in candidate_ids if c not in seen]
        rankings[qid] = ranked + missing
        if (i + 1) % 25 == 0 or (i + 1) == len(queries):
            log.info("    %d / %d  (%.1fs)", i + 1, len(queries), time.time() - t0)
    log.info("  Done %s in %.1fs", label, time.time() - t0)
    return rankings


def save_ranking_file(rankings: dict, path: Path, top_n: int = None):
    """Save a rankings dict to a JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    out = {qid: ranked[:top_n] if top_n else ranked
           for qid, ranked in rankings.items()}
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    log.info("  Saved %s  (%d queries)", path.name, len(out))


# ===================================================================
# Data helpers
# ===================================================================

def load_split(split):
    qdf = pd.read_parquet(DATA_DIR / f"{split}_queries-00000-of-00001.parquet")
    cdf = pd.read_parquet(DATA_DIR / f"{split}_candidates-00000-of-00001.parquet")
    return qdf.to_dict("records"), cdf.to_dict("records")

def doc_text(doc):
    t = doc["text"]
    return t if isinstance(t, str) else " ".join(str(s) for s in t)

STATUTE_RE = re.compile(
    r"(?:Section|Sections|S\.)\s+(\d+[A-Z]?(?:\s*[\(/]\s*\d+\s*[\)/])?)"
    r"(?:\s+(?:of|of the)\s+)?"
    r"((?:Indian Penal Code|IPC|Cr\.?\s*P\.?\s*C\.?|C\.?\s*P\.?\s*C\.?|"
    r"Constitution|Evidence Act|Contract Act|Companies Act|"
    r"[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+Act(?:,?\s*\d{4})?))?",
    re.IGNORECASE,
)

def extract_statutes(text):
    out = set()
    for m in STATUTE_RE.finditer(text):
        sec = m.group(1).strip()
        act = (m.group(2) or "").strip()
        out.add(f"S.{sec}" + (f" {act}" if act else ""))
    return out

def expand_query(text, n_tokens=80):
    tokens = re.findall(r"[a-zA-Z]{3,}", text)
    freq   = defaultdict(int)
    for t in tokens:
        freq[t.lower()] += 1
    top = [t for t, _ in sorted(freq.items(), key=lambda x: -x[1])[:n_tokens]]
    return " ".join(list(extract_statutes(text)) + top) + " " + text[:3000]


# ===================================================================
# Triple store + memory-safe sparse index
# ===================================================================

def triple_to_text(h, r, t):
    rel_map = {
        "CITES_PRECEDENT": "cites precedent", "APPLIES_STATUTE": "applies statute",
        "PRESIDED_BY": "presided by judge",    "HEARD_IN": "heard in court",
        "REFERENCES_CASE": "references case",  "SHARES_STATUTE": "shares statute with",
        "SHARES_JUDGE": "shares judge with",
    }
    return f"{h} {rel_map.get(r, r)} {t}"

class TripleStore:
    def __init__(self, graph_path):
        log.info("[Stage 1] Loading KG...")
        self.G = nx.read_graphml(str(graph_path))
        log.info("  KG: %d nodes, %d edges", self.G.number_of_nodes(), self.G.number_of_edges())
        self.triples = [
            (u, r.strip(), v)
            for u, v, data in self.G.edges(data=True)
            for r in data.get("type", "RELATED").split(",")
            if r.strip() in ("CITES_PRECEDENT", "APPLIES_STATUTE", "PRESIDED_BY",
                             "HEARD_IN", "REFERENCES_CASE")
        ]
        log.info("  %d triples", len(self.triples))

    def get_entity_context(self, case_id):
        if case_id not in self.G:
            return []
        return [
            (case_id, r.strip(), nbr)
            for nbr in self.G.neighbors(case_id)
            for r in self.G[case_id][nbr].get("type", "RELATED").split(",")
        ]


class SparseTripleIndex:
    """
    Memory-safe replacement for FAISSTripleIndex.
    Uses scipy sparse matrix + dot product instead of dense FAISS.
    72k triples x 15k vocab (sparse) ~50 MB instead of ~14 GB.
    """
    MAX_FEATURES = 15_000

    def __init__(self, triples):
        log.info("[Stage 1] Building sparse triple index for %d triples "
                 "(max_features=%d)...", len(triples), self.MAX_FEATURES)
        self.triples = triples
        self.vec = TfidfVectorizer(
            max_features=self.MAX_FEATURES,
            sublinear_tf=True,
            ngram_range=(1, 2),
        )
        self.matrix = normalize(
            self.vec.fit_transform([triple_to_text(*t) for t in triples]),
            norm="l2",
        )
        log.info("  Sparse triple index: %s, nnz=%d",
                 self.matrix.shape, self.matrix.nnz)

    def retrieve(self, query_text: str, top_k: int = 30):
        qvec    = normalize(self.vec.transform([query_text]), norm="l2")
        scores  = (self.matrix @ qvec.T).toarray().ravel()
        top_idx = np.argpartition(scores, -min(top_k, len(scores)))[-top_k:]
        top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]
        return [(self.triples[i], float(scores[i])) for i in top_idx]


# ===================================================================
# Retrievers
# ===================================================================

class KGRetriever:
    def __init__(self, G):
        self.G = G

    def retrieve(self, query_id, candidate_ids):
        if query_id not in self.G:
            return []
        scores, q_nbrs = {}, set(self.G.neighbors(query_id))
        for cid in candidate_ids:
            if cid not in self.G:
                continue
            s      = 0.0
            shared = q_nbrs & set(self.G.neighbors(cid))
            for n in shared:
                nt = self.G.nodes[n].get("type", "")
                if   nt == "Statute": s += 2.0
                elif nt == "Judge":   s += 1.5
                elif nt == "Court":   s += 0.5
            if self.G.has_edge(query_id, cid):
                etype = self.G[query_id][cid].get("type", "")
                if "SHARES_STATUTE" in etype:
                    s += 3.0 * int(self.G[query_id][cid].get("shared_count", 1))
                if "SHARES_JUDGE" in etype:
                    s += 2.0 * int(self.G[query_id][cid].get("shared_count", 1))
            qc = self.G.nodes.get(query_id, {}).get("community", -1)
            cc = self.G.nodes.get(cid,      {}).get("community", -2)
            if qc == cc and qc != -1:
                s += 1.0
            if s > 0:
                scores[cid] = s
        return [c for c, _ in sorted(scores.items(), key=lambda x: -x[1])]


class RAGRetriever:
    def __init__(self, queries, candidates):
        log.info("[RAG-BM25] Building index over %d candidates...", len(candidates))
        self.cand_ids    = [str(c["id"]) for c in candidates]
        self.query_texts = {str(q["id"]): expand_query(doc_text(q)) for q in queries}
        self.bm25        = BM25VectorizerIndex([doc_text(c)[:6000] for c in candidates])

    def retrieve(self, query_id, candidate_ids):
        s, i    = self.bm25.search(self.query_texts.get(query_id, ""), top_k=len(self.cand_ids))
        cid_set = set(candidate_ids)
        return [self.cand_ids[idx] for _, idx in zip(s, i)
                if idx >= 0 and self.cand_ids[idx] in cid_set]


class RAGLLMRetriever:
    def __init__(self, queries, candidates, llm=None, rerank_top_n=60):
        log.info("[RAG+LLM] Building index...")
        self.llm, self.rerank_top_n = llm, rerank_top_n
        self.cand_ids    = [str(c["id"]) for c in candidates]
        self.query_ids   = [str(q["id"]) for q in queries]
        self.cand_texts  = {str(c["id"]): doc_text(c)[:6000] for c in candidates}
        self.query_texts = {str(q["id"]): doc_text(q)[:6000] for q in queries}
        self.bm25        = BM25VectorizerIndex([self.cand_texts[c] for c in self.cand_ids])

        self.cand_statutes  = {c: extract_statutes(t) for c, t in self.cand_texts.items()}
        self.query_statutes = {q: extract_statutes(t) for q, t in self.query_texts.items()}
        stat_df = defaultdict(int)
        for ss in self.cand_statutes.values():
            for s in ss: stat_df[s] += 1
        N = len(self.cand_ids)
        self.stat_idf = {s: math.log((N + 1) / (df + 1)) for s, df in stat_df.items()}

        def _kw(text, n=60):
            freq = defaultdict(int)
            for t in re.findall(r"[a-zA-Z]{4,}", text.lower()): freq[t] += 1
            return {t for t, _ in sorted(freq.items(), key=lambda x: -x[1])[:n]}

        self.cand_kw  = {c: _kw(t) for c, t in self.cand_texts.items()}
        self.query_kw = {q: _kw(t) for q, t in self.query_texts.items()}
        cre = re.compile(r"\b\d{4}\s*\(\d+\)\s*[A-Z]+\s+\d+\b|\b[A-Z]+\s+No\.\s*\d+\b", re.I)
        self.cand_citeno  = {c: set(cre.findall(t.upper())) for c, t in self.cand_texts.items()}
        self.query_citeno = {q: set(cre.findall(t.upper())) for q, t in self.query_texts.items()}
        log.info("  RAG+LLM index built.")

    def _rerank_score(self, qid, cid, base):
        score  = base
        qs, cs = self.query_statutes.get(qid, set()), self.cand_statutes.get(cid, set())
        if qs | cs:
            score += (len(qs & cs) / len(qs | cs)) * 0.40 + \
                     sum(self.stat_idf.get(s, 1.) for s in qs & cs) * 0.01
        qk, ck = self.query_kw.get(qid, set()), self.cand_kw.get(cid, set())
        if qk | ck:
            score += (len(qk & ck) / len(qk | ck)) * 0.20
        if self.query_citeno.get(qid, set()) & self.cand_citeno.get(cid, set()):
            score += 0.30
        return score

    def retrieve(self, query_id, candidate_ids):
        s, i    = self.bm25.search(expand_query(self.query_texts.get(query_id, "")),
                                   top_k=len(self.cand_ids))
        cid_set = set(candidate_ids)
        initial = [(self.cand_ids[idx], float(sc)) for sc, idx in zip(s, i)
                   if idx >= 0 and self.cand_ids[idx] in cid_set]
        top, rest = initial[:self.rerank_top_n], initial[self.rerank_top_n:]
        reranked  = sorted([(c, self._rerank_score(query_id, c, sc)) for c, sc in top],
                           key=lambda x: -x[1])
        return [c for c, _ in reranked] + [c for c, _ in rest]


class LLMReranker:
    def __init__(self, api_key, model="llama-3.3-70b-versatile"):
        self.api_key, self.model, self._client = api_key, model, None

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(api_key=self.api_key, base_url="https://api.groq.com/openai/v1")
        return self._client

    def predict(self, prompt):
        try:
            raw = self._get_client().chat.completions.create(
                model=self.model, messages=[{"role": "user", "content": prompt}],
                max_tokens=500, temperature=0.1,
            ).choices[0].message.content.strip()
            s, e = raw.find("["), raw.rfind("]")
            if s != -1 and e != -1:
                return json.loads(raw[s:e + 1])
        except Exception as ex:
            log.debug("LLM failed: %s", ex)
        return []


class RAKGLLMRetriever:
    def __init__(self, triple_store, sparse_triple_index, queries, candidates, llm=None):
        self.store  = triple_store
        self.faiss  = sparse_triple_index
        self.llm    = llm
        self.cand_ids    = [str(c["id"]) for c in candidates]
        self.query_ids   = [str(q["id"]) for q in queries]
        cand_id_set      = set(self.cand_ids)
        self.cand_texts  = {str(c["id"]): doc_text(c)[:6000] for c in candidates}
        self.query_texts = {str(q["id"]): doc_text(q)[:6000] for q in queries}
        self.bm25        = BM25VectorizerIndex([self.cand_texts[c] for c in self.cand_ids])
        log.info("[RA-KG-LLM] Building entity maps...")
        G = self.store.G
        self.case_entities = {}
        for cid in set(self.query_ids) | cand_id_set:
            if cid not in G: continue
            ents = {"Statute": set(), "Judge": set(), "Court": set()}
            for nbr in G.neighbors(cid):
                nt = G.nodes[nbr].get("type", "")
                if nt in ents: ents[nt].add(nbr)
            self.case_entities[cid] = ents
        self.statute_to_cases = defaultdict(set)
        for cid in cand_id_set:
            for s in self.case_entities.get(cid, {}).get("Statute", set()):
                self.statute_to_cases[s].add(cid)
        log.info("  Entity maps: %d cases", len(self.case_entities))

    def retrieve(self, query_id, candidate_ids):
        cid_set = set(candidate_ids)
        scores  = defaultdict(float)
        s, i    = self.bm25.search(expand_query(self.query_texts.get(query_id, "")),
                                   top_k=len(self.cand_ids))
        for sc, idx in zip(s, i):
            if idx < 0: continue
            cid = self.cand_ids[idx]
            if cid in cid_set: scores[cid] += float(sc) * 5.0

        ctx  = self.store.get_entity_context(query_id)
        toks = [t for _, _, t in ctx
                if self.store.G.nodes.get(t, {}).get("type", "") in ("Statute", "Judge", "Court")]
        for (h, r, t), sim in self.faiss.retrieve(
                "cites precedent " + " ".join(toks[:15]), top_k=40):
            if r == "CITES_PRECEDENT":
                if t in cid_set: scores[t] += sim * 4.0
                if h in cid_set: scores[h] += sim * 2.0
            elif r == "APPLIES_STATUTE":
                for cid in self.statute_to_cases.get(t, set()):
                    if cid in cid_set: scores[cid] += sim * 3.0

        qe = self.case_entities.get(query_id, {})
        qs, qj = qe.get("Statute", set()), qe.get("Judge", set())
        for cid in candidate_ids:
            ce = self.case_entities.get(cid)
            if not ce: continue
            cs, cj = ce.get("Statute", set()), ce.get("Judge", set())
            if qs | cs: scores[cid] += len(qs & cs) / len(qs | cs) * 2.0
            if qj:      scores[cid] += len(qj & cj) / len(qj) * 1.0

        return [c for c, _ in sorted(scores.items(), key=lambda x: -x[1])]


class HybridRetriever:
    WEIGHTS = {"KG": 0.5, "RAG-BM25": 2.5, "RAG+LLM": 2.0, "RA-KG-LLM": 1.5}
    K = 60

    def __init__(self, retrievers):
        self.retrievers = retrievers

    def retrieve(self, query_id, candidate_ids):
        combined = defaultdict(float)
        for name, retr in self.retrievers.items():
            w = self.WEIGHTS.get(name, 1.0)
            for rank, cid in enumerate(retr.retrieve(query_id, candidate_ids)):
                combined[cid] += w / (self.K + rank + 1)
        return [c for c, _ in sorted(combined.items(), key=lambda x: -x[1])]


# ===================================================================
# Plotting
# ===================================================================

PALETTE      = {"KG": "#2563EB", "RAG-BM25": "#16A34A", "RAG+LLM": "#D97706",
                "RA-KG-LLM": "#DC2626", "Hybrid-RRF": "#7C3AED"}
METHOD_ORDER = ["KG", "RAG-BM25", "RAG+LLM", "RA-KG-LLM", "Hybrid-RRF"]


def _style_ax(ax, title, xlabel, ylabel):
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.tick_params(labelsize=10)


def plot_metrics_bar(res, k=10, save_path=None):
    metrics = [f"P@{k}", f"R@{k}", f"F1@{k}", f"nDCG@{k}", "MAP", "Hits@1"]
    methods = [m for m in METHOD_ORDER if m in res]
    x, w    = np.arange(len(metrics)), 0.75 / len(methods)
    fig, ax = plt.subplots(figsize=(13, 5))
    for i, method in enumerate(methods):
        vals   = [res[method].get(m, 0) for m in metrics]
        offset = (i - len(methods) / 2 + 0.5) * w
        bars   = ax.bar(x + offset, vals, w * 0.9, label=method,
                        color=PALETTE.get(method, "#888"), edgecolor="white", linewidth=0.6)
        for bar, v in zip(bars, vals):
            if v > 0.02:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.004,
                        f"{v:.3f}", ha="center", va="bottom", fontsize=6.5, rotation=90)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=11)
    ax.set_ylim(0, min(1.05, ax.get_ylim()[1] * 1.25))
    _style_ax(ax, f"Metric Comparison  (K={k})", "Metric", "Score")
    ax.legend(loc="upper right", framealpha=0.9, fontsize=9)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        log.info("  Saved: %s", save_path)
    plt.close(fig)


def plot_ndcg_vs_k(res, k_values, save_path=None):
    methods   = [m for m in METHOD_ORDER if m in res]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, base in zip(axes, ["nDCG", "F1"]):
        for method in methods:
            vals = [res[method].get(f"{base}@{k}", 0) for k in k_values]
            ax.plot(k_values, vals, marker="o", linewidth=2.2, markersize=7,
                    label=method, color=PALETTE.get(method, "#888"))
            for k, v in zip(k_values, vals):
                ax.annotate(f"{v:.3f}", (k, v), textcoords="offset points",
                            xytext=(0, 7), ha="center", fontsize=7.5,
                            color=PALETTE.get(method, "#888"))
        _style_ax(ax, f"{base}@K across Methods", "K", f"{base}@K")
        ax.set_xticks(k_values)
        ax.legend(loc="lower right", framealpha=0.9, fontsize=9)
        ax.set_ylim(0, min(1.05, ax.get_ylim()[1] + 0.1))
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        log.info("  Saved: %s", save_path)
    plt.close(fig)


def plot_radar(res, k=10, save_path=None):
    metrics = [f"P@{k}", f"R@{k}", f"F1@{k}", f"nDCG@{k}", "MAP", "Hits@1"]
    methods = [m for m in METHOD_ORDER if m in res]
    N       = len(metrics)
    angles  = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist() + [0]
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, fontsize=11)
    for method in methods:
        vals = [res[method].get(m, 0) for m in metrics] + [res[method].get(metrics[0], 0)]
        ax.plot(angles, vals, linewidth=2, color=PALETTE.get(method, "#888"), label=method)
        ax.fill(angles, vals, alpha=0.08, color=PALETTE.get(method, "#888"))
    ax.set_title(f"Method Radar  (K={k})", fontsize=14, fontweight="bold", pad=25)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), framealpha=0.9, fontsize=10)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        log.info("  Saved: %s", save_path)
    plt.close(fig)


def plot_pr_bubble(res, k=10, save_path=None):
    methods = [m for m in METHOD_ORDER if m in res]
    fig, ax = plt.subplots(figsize=(8, 6))
    for method in methods:
        p, r = res[method].get(f"P@{k}", 0), res[method].get(f"R@{k}", 0)
        ms   = res[method].get("MAP", 0)
        ax.scatter(r, p, s=300 + ms * 3000, color=PALETTE.get(method, "#888"),
                   alpha=0.75, edgecolors="white", linewidth=1.5, zorder=3)
        ax.annotate(f"{method}\nMAP={ms:.3f}", (r, p), textcoords="offset points",
                    xytext=(8, 4), fontsize=8.5, fontweight="bold",
                    color=PALETTE.get(method, "#333"))
    for f1i in [0.1, 0.2, 0.3, 0.4, 0.5]:
        rc   = np.linspace(0.01, 0.99, 200)
        pc   = f1i * rc / (2 * rc - f1i)
        mask = (pc > 0) & (pc < 1)
        ax.plot(rc[mask], pc[mask], "--", color="#ccc", linewidth=0.8)
        idx = len(rc[mask]) // 2
        ax.text(rc[mask][idx], pc[mask][idx], f"F1={f1i}", fontsize=7,
                color="#999", ha="center")
    _style_ax(ax, f"Precision–Recall  (K={k})  bubble ∝ MAP",
              f"Recall@{k}", f"Precision@{k}")
    ax.set_xlim(0, min(1.1, ax.get_xlim()[1] + 0.05))
    ax.set_ylim(0, min(1.1, ax.get_ylim()[1] + 0.05))
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        log.info("  Saved: %s", save_path)
    plt.close(fig)


def save_all_plots(res, k_values, split):
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    mid_k = k_values[len(k_values) // 2]
    plot_metrics_bar(res, k=mid_k, save_path=PLOTS_DIR / f"01_metrics_bar_{split}_k{mid_k}.png")
    plot_ndcg_vs_k(res, k_values,  save_path=PLOTS_DIR / f"02_ndcg_f1_vs_k_{split}.png")
    plot_radar(res, k=mid_k,       save_path=PLOTS_DIR / f"03_radar_{split}_k{mid_k}.png")
    plot_pr_bubble(res, k=mid_k,   save_path=PLOTS_DIR / f"04_pr_bubble_{split}_k{mid_k}.png")
    log.info("Plots saved → %s", PLOTS_DIR)


# ===================================================================
# Print results table
# ===================================================================

def print_comparison(all_results, k_values, split, nq, nc):
    print("\n" + "=" * 115)
    print(f"  PCR Retrieval v2 — {split} split  ({nq} queries, {nc} candidates)")
    print("=" * 115)
    hdr = f"{'Method':<25} {'Hits@1':>8} {'MAP':>8}"
    for k in k_values:
        hdr += f"  {'P@' + str(k):>7} {'R@' + str(k):>7} {'F1@' + str(k):>7} {'nDCG@' + str(k):>8}"
    print(hdr)
    print("-" * len(hdr))
    for method in METHOD_ORDER:
        if method not in all_results: continue
        m   = all_results[method]
        row = f"{method:<25} {m['Hits@1']:>8.4f} {m['MAP']:>8.4f}"
        for k in k_values:
            row += f"  {m[f'P@{k}']:>7.4f} {m[f'R@{k}']:>7.4f} {m[f'F1@{k}']:>7.4f} {m[f'nDCG@{k}']:>8.4f}"
        print(row)
    print("-" * len(hdr))
    best      = max(all_results.items(), key=lambda x: x[1]["MAP"])
    best_f1   = max(all_results.items(), key=lambda x: x[1][f"F1@{max(k_values)}"])
    best_ndcg = max(all_results.items(), key=lambda x: x[1][f"nDCG@{max(k_values)}"])
    print(f"\n  Best MAP      : {best[0]}  ({best[1]['MAP']:.4f})")
    print(f"  Best F1@{max(k_values)}    : {best_f1[0]}  ({best_f1[1][f'F1@{max(k_values)}']:.4f})")
    print(f"  Best nDCG@{max(k_values)}  : {best_ndcg[0]}  ({best_ndcg[1][f'nDCG@{max(k_values)}']:.4f})")
    print("\n  Pairwise Δ-MAP (row − col):")
    methods = [m for m in METHOD_ORDER if m in all_results]
    print("  " + f"{'':25}" + "".join(f"{m:>14}" for m in methods))
    for m1 in methods:
        row = f"  {m1:<25}"
        for m2 in methods:
            row += f"  {all_results[m1]['MAP'] - all_results[m2]['MAP']:>+.4f}    "
        print(row)
    print()


# ===================================================================
# ★  NEW: build & save top-5 RAG-only rankings for EVERY query
# ===================================================================

TOP_5 = 5

def build_top5_rag_rankings(all_rankings: dict, queries: list) -> dict:
    """
    Returns an OrderedDict keyed by query-id in the original dataset order,
    where each value is:
        { "RAG-BM25": [c1, c2, c3, c4, c5],
          "RAG+LLM":  [c1, c2, c3, c4, c5] }
    covering ALL queries (not just a sample).
    """
    from collections import OrderedDict
    # Iterate queries in their original dataset order — OrderedDict
    # guarantees that order is preserved exactly when serialised to JSON.
    output = OrderedDict()
    for q in queries:
        qid = str(q["id"])
        output[qid] = {
            model: all_rankings[model].get(qid, [])[:TOP_5]
            for model in RAG_MODELS
        }
    return output


def save_top5_rag_rankings(top5: dict, split: str) -> Path:
    """Persist the top-5 dict and return the path.

    sort_keys=False is set explicitly so json.dump never reorders the
    query keys — they stay in the original dataset order.
    """
    path = RANK_DIR / f"per_query_rag_top5_{split}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(top5, f, indent=2, sort_keys=False)
    log.info("  Saved top-5 RAG rankings → %s  (%d queries)", path.name, len(top5))
    return path


def print_top5_rag_rankings(top5: dict):
    """Pretty-print ALL queries' top-5 rankings for RAG models."""
    print("\n" + "=" * 70)
    print(f"  Per-Query Top-{TOP_5} Rankings — RAG Models Only  ({len(top5)} queries)")
    print("=" * 70)
    for qid, model_ranks in top5.items():
        print(json.dumps({qid: model_ranks}, indent=2))
    print("=" * 70)


# ===================================================================
# Main
# ===================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split",    default="dev", choices=["dev", "test", "train"])
    parser.add_argument("--k",        nargs="+", type=int, default=[5, 10, 20])
    parser.add_argument("--top-n",    type=int, default=None,
                        help="Truncate full saved rankings to top-N per query (default: save all)")
    parser.add_argument("--groq-key", default=os.getenv("GROQ_API_KEY", ""))
    args = parser.parse_args()

    log.info("=" * 60)
    log.info("RA-KG-LLM v2 — %s split", args.split)
    log.info("=" * 60)

    queries, candidates = load_split(args.split)
    candidate_ids       = [str(c["id"]) for c in candidates]
    log.info("Data: %d queries, %d candidates", len(queries), len(candidates))

    graph_path = OUTPUT_DIR / "pcr_knowledge_graph.graphml"
    if not graph_path.exists():
        log.error("KG not found at %s", graph_path)
        return

    RANK_DIR.mkdir(parents=True, exist_ok=True)

    all_rankings: dict = {}
    all_results:  dict = {}
    timings:      dict = {}
    llm = LLMReranker(args.groq_key) if args.groq_key else None

    # ── 1. KG ──────────────────────────────────────────────────────
    log.info("\n── 1. KG ──")
    G  = nx.read_graphml(str(graph_path))
    kg = KGRetriever(G)
    t0 = time.time()
    all_rankings["KG"] = run_and_collect(kg.retrieve, queries, candidate_ids, "KG")
    timings["KG"]      = time.time() - t0
    all_results["KG"]  = evaluate_from_rankings(all_rankings["KG"], queries, args.k)
    log.info("  MAP=%.4f  (%.1fs)  [no JSON saved — not a RAG model]",
             all_results["KG"]["MAP"], timings["KG"])

    # ── 2. RAG-BM25 ── saves full + top-5 JSON ─────────────────────
    log.info("\n── 2. RAG-BM25 ──")
    rag = RAGRetriever(queries, candidates)
    t0  = time.time()
    all_rankings["RAG-BM25"] = run_and_collect(rag.retrieve, queries, candidate_ids, "RAG-BM25")
    timings["RAG-BM25"]      = time.time() - t0
    save_ranking_file(all_rankings["RAG-BM25"], RANK_DIR / "rag_bm25_rankings.json", args.top_n)
    all_results["RAG-BM25"]  = evaluate_from_rankings(all_rankings["RAG-BM25"], queries, args.k)
    log.info("  MAP=%.4f  (%.1fs)", all_results["RAG-BM25"]["MAP"], timings["RAG-BM25"])

    # ── 3. RAG+LLM ── saves full + top-5 JSON ──────────────────────
    log.info("\n── 3. RAG+LLM ──")
    rag_llm = RAGLLMRetriever(queries, candidates, llm=llm)
    t0      = time.time()
    all_rankings["RAG+LLM"] = run_and_collect(rag_llm.retrieve, queries, candidate_ids, "RAG+LLM")
    timings["RAG+LLM"]      = time.time() - t0
    save_ranking_file(all_rankings["RAG+LLM"], RANK_DIR / "rag_llm_rankings.json", args.top_n)
    all_results["RAG+LLM"]  = evaluate_from_rankings(all_rankings["RAG+LLM"], queries, args.k)
    log.info("  MAP=%.4f  (%.1fs)", all_results["RAG+LLM"]["MAP"], timings["RAG+LLM"])

    # ── 4. RA-KG-LLM ───────────────────────────────────────────────
    log.info("\n── 4. RA-KG-LLM ──")
    store = TripleStore(graph_path)
    fidx  = SparseTripleIndex(store.triples)
    ra_kg = RAKGLLMRetriever(store, fidx, queries, candidates, llm=llm)
    t0    = time.time()
    all_rankings["RA-KG-LLM"] = run_and_collect(ra_kg.retrieve, queries, candidate_ids, "RA-KG-LLM")
    timings["RA-KG-LLM"]      = time.time() - t0
    all_results["RA-KG-LLM"]  = evaluate_from_rankings(all_rankings["RA-KG-LLM"], queries, args.k)
    log.info("  MAP=%.4f  (%.1fs)  [no JSON saved — not a RAG model]",
             all_results["RA-KG-LLM"]["MAP"], timings["RA-KG-LLM"])

    # ── 5. Hybrid-RRF ──────────────────────────────────────────────
    log.info("\n── 5. Hybrid-RRF ──")
    hybrid = HybridRetriever({"KG": kg, "RAG-BM25": rag, "RAG+LLM": rag_llm, "RA-KG-LLM": ra_kg})
    t0     = time.time()
    all_rankings["Hybrid-RRF"] = run_and_collect(hybrid.retrieve, queries, candidate_ids, "Hybrid-RRF")
    timings["Hybrid-RRF"]      = time.time() - t0
    all_results["Hybrid-RRF"]  = evaluate_from_rankings(all_rankings["Hybrid-RRF"], queries, args.k)
    log.info("  MAP=%.4f  (%.1fs)  [no JSON saved — not a RAG model]",
             all_results["Hybrid-RRF"]["MAP"], timings["Hybrid-RRF"])

    # ── ★ Top-5 RAG-only per-query JSON (ALL queries) ──────────────
    log.info("\n── Building RAG-only top-5 per-query JSON (ALL queries) ──")
    top5_rag = build_top5_rag_rankings(all_rankings, queries)
    top5_path = save_top5_rag_rankings(top5_rag, args.split)

    # ── Terminal: print ALL queries top-5 RAG rankings ─────────────
    print_top5_rag_rankings(top5_rag)

    # ── Evaluation table (all 5 models for comparison) ─────────────
    print_comparison(all_results, args.k, args.split, len(queries), len(candidates))

    print("  Wall-clock times:")
    for m, t in timings.items():
        print(f"    {m:<25} {t:>7.1f}s")

    # ── Save full metrics JSON (all models) ────────────────────────
    OUTPUT_DIR.mkdir(exist_ok=True)
    with open(OUTPUT_DIR / f"ra_kg_llm_v2_{args.split}.json", "w") as f:
        json.dump({"results": all_results, "timings": timings,
                   "split": args.split, "k_values": args.k}, f, indent=2)

    # ── Plots ──────────────────────────────────────────────────────
    log.info("\nGenerating plots...")
    save_all_plots(all_results, args.k, args.split)

    # ── File summary ───────────────────────────────────────────────
    log.info("\n── Output files (rankings) ──")
    for fp in sorted(RANK_DIR.glob("*.json")):
        log.info("  %-55s  %.1f KB", fp.name, fp.stat().st_size / 1024)
    log.info("\n── Output files (plots) ──")
    for fp in sorted(PLOTS_DIR.glob("*.png")):
        log.info("  %-55s  %.1f KB", fp.name, fp.stat().st_size / 1024)

    log.info("\nDone. Top-5 RAG rankings for all queries → %s", top5_path)


if __name__ == "__main__":
    main()
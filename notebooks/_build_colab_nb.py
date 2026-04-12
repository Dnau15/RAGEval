"""Builder for `notebooks/run_on_colab.ipynb`.

Run with:
    python notebooks/_build_colab_nb.py

This emits a self-contained Colab notebook that:
  * Mounts Drive into `/content/drive/MyDrive/RAGEval`.
  * Bootstraps every "seed" CSV the experiment cells expect (so a fresh
    Drive folder is enough -- no manual file uploads required).
  * Runs Phase B.1 (BioASQ retrieval), B.2 (BGE cross-encoder reranker
    + optional MedCPT-CE), B.3 (query router), B.4 (MIRAGE/MedRAG)
    end-to-end, with disk caching keyed off `OUT/cache`.
"""

from __future__ import annotations

import json
from pathlib import Path

CELLS = []


def md(text: str) -> None:
    CELLS.append({"cell_type": "markdown", "metadata": {}, "source": text})


def code(text: str) -> None:
    CELLS.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text,
    })


# ---------------------------------------------------------------------------
# 0. Header
# ---------------------------------------------------------------------------
md("""# RAGEval -- Round-2 Feedback: All-in-One Colab Notebook

This notebook reproduces every Round-2 feedback experiment in a single
top-to-bottom run on a Colab T4 GPU. Upload it to Colab, attach a T4 (or
better), and execute every cell sequentially.

**What it produces (under `<ROOT>/notebooks/results/feedback2/tables/`):**

| File | Source | Used by main.tex |
|---|---|---|
| `nfcorpus_canonical.csv` | Phase A bootstrap | Table 3 |
| `multi_dataset_ndcg10_v2.csv` | Phase A bootstrap + Phase B.1 backfill | Table 4 |
| `n_required.csv` | Phase A bootstrap | Table (n_required) |
| `bioasq_paired_bootstrap.csv` | Phase B.1 | narrative |
| `reranker_ndcg10.csv` | Phase B.2 (BGE-reranker-base + optional MedCPT-CE) | Table 5 |
| `router_test.csv`, `router_feature_importance.csv`, `router_train_test_gap.csv` | Phase B.3 | Tables 6, 7 |
| `mirage_accuracy.csv` (+ `cache/mirage_predictions.parquet`) | Phase B.4 | Table 8 |

**Default output root:** `/content/drive/MyDrive/RAGEval`. Change `ROOT` in
the setup cell if you keep the project elsewhere.

**Llama-3.2-3B-Instruct is gated.** Run `huggingface-cli login` *before*
B.4, or B.4 will silently skip Llama and report only flan-t5-base.
""")


# ---------------------------------------------------------------------------
# 1. Pip install -- single stage. The earlier multi-stage gymnastics were
#    only needed to undo damage that pylate's resolver did to the
#    transformers stack; with pylate removed, a normal install is fine.
# ---------------------------------------------------------------------------
code("""!pip install -q \\
    "beir==2.0.0" "bm25s==0.2.6" "datasets>=2.18.0" \\
    "scikit-learn>=1.3.0" "lightgbm>=4.0.0" "scipy>=1.11.0" \\
    "faiss-cpu>=1.7.4" "rank-bm25>=0.2.2" "rouge-score>=0.1.2" \\
    "nltk>=3.8.0" "numpy>=1.24.0" "pandas>=2.0.0" "psutil>=5.9.0" \\
    "tqdm>=4.65.0" "sentencepiece" \\
    "transformers>=4.45.0,<5" "sentence-transformers>=3.0.0,<5" \\
    "bitsandbytes>=0.43.0" "accelerate>=0.34.0"
print("\\nDependencies installed. Continue with cell 2.")
""")


# ---------------------------------------------------------------------------
# 2. Mount Drive + ROOT
# ---------------------------------------------------------------------------
code("""# Mount Drive (Colab) and pin a single ROOT for all artefacts.
import os
from pathlib import Path

IN_COLAB = os.path.exists("/content")

if IN_COLAB:
    from google.colab import drive
    drive.mount("/content/drive", force_remount=False)
    ROOT = Path("/content/drive/MyDrive/RAGEval")
else:
    # Local fallback: keep the project in the repo working tree.
    ROOT = Path.cwd()
    if (ROOT / "notebooks").exists() is False and (ROOT.parent / "notebooks").exists():
        ROOT = ROOT.parent

ROOT.mkdir(parents=True, exist_ok=True)
OUT = ROOT / "notebooks" / "results" / "feedback2"
DATASETS_DIR = ROOT / "notebooks" / "datasets"
CACHE_DIR = OUT / "cache"
for sub in ["tables", "datasets", "figures", "cache"]:
    (OUT / sub).mkdir(parents=True, exist_ok=True)
DATASETS_DIR.mkdir(parents=True, exist_ok=True)

print("ROOT:", ROOT)
print("OUT :", OUT)
""")


# ---------------------------------------------------------------------------
# 3. Imports + device
# ---------------------------------------------------------------------------
code("""# Core imports + device handle.
import gc, json, math, os, pickle, random, time, importlib, subprocess, sys, re
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

import transformers
print(f"transformers {transformers.__version__} -- OK")

from beir import util as beir_util  # NFCorpus / TREC-COVID / SciFact / ArguAna downloads
from beir.datasets.data_loader import GenericDataLoader

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)


def cuda_gc():
    \"\"\"Free Python refs + reclaim CUDA cache fragment (helps T4 Colab OOMs).\"\"\"
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# T4-safe default: cross-encoder forward is heavy at fp32 × batch 32.
# Set env RAGEVAL_RERANK_BATCH=16–32 if you use a larger GPU and want speed.
RERANK_BATCH = int(os.environ.get("RAGEVAL_RERANK_BATCH", "8"))
print("RERANK_BATCH (cross-encoder predict):", RERANK_BATCH)


def _pip(pkgs):
    \"\"\"Idempotent installer used by individual cells that need extras.\"\"\"
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q"] + pkgs)
""")


# ---------------------------------------------------------------------------
# 4. Metric + retrieval helpers
# ---------------------------------------------------------------------------
code("""# Metric helpers (mirror final.ipynb / cell 13) -----------------------------
def compute_ndcg(ranked_docs, qrel, k):
    dcg = 0.0
    for i, did in enumerate(ranked_docs[:k]):
        rel = qrel.get(did, 0)
        dcg += (2 ** rel - 1) / np.log2(i + 2)
    ideal = sorted(qrel.values(), reverse=True)[:k]
    idcg = sum((2 ** r - 1) / np.log2(i + 2) for i, r in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def compute_recall(ranked_docs, qrel, k, threshold=1):
    relevant = {d for d, r in qrel.items() if r >= threshold}
    if not relevant:
        return 0.0
    found = sum(1 for d in ranked_docs[:k] if d in relevant)
    return found / len(relevant)


def compute_precision(ranked_docs, qrel, k, threshold=1):
    relevant = {d for d, r in qrel.items() if r >= threshold}
    if k == 0:
        return 0.0
    found = sum(1 for d in ranked_docs[:k] if d in relevant)
    return found / k


def evaluate_retriever(results, qrels, k_values=(1, 3, 5, 10)):
    per_query = {}
    for qid in qrels:
        if qid not in results:
            continue
        ranking = [d for d, _ in sorted(results[qid].items(), key=lambda x: -x[1])]
        m = {}
        for k in k_values:
            m[f"nDCG@{k}"]  = compute_ndcg(ranking, qrels[qid], k)
            m[f"Recall@{k}"] = compute_recall(ranking, qrels[qid], k)
            m[f"P@{k}"]      = compute_precision(ranking, qrels[qid], k)
        per_query[qid] = m
    if not per_query:
        return {"aggregate": {}, "per_query": {}, "loss_array": np.array([])}
    keys = list(next(iter(per_query.values())).keys())
    agg = {k: float(np.mean([m[k] for m in per_query.values()])) for k in keys}
    loss = np.array([1.0 - m["nDCG@10"] for m in per_query.values()])
    return {"aggregate": agg, "per_query": per_query, "loss_array": loss}


# Retrieval helpers --------------------------------------------------------
def _prep_corpus(corpus):
    ids = list(corpus.keys())
    texts = [(corpus[d].get("title", "") + " " + corpus[d].get("text", "")).strip() for d in ids]
    return ids, texts


def _bm25_run(doc_ids, doc_texts, queries, top_k=100):
    \"\"\"BM25 over `queries`. Forces single-threaded retrieval to avoid the
    bm25s 0.2.x multiprocessing bug on Colab kernels (worker pool returns
    zero items -> `not enough values to unpack (expected 2, got 0)`).\"\"\"
    import bm25s
    if not queries:
        return {}
    corpus_tok = bm25s.tokenize(doc_texts, stopwords="en", show_progress=False)
    bm = bm25s.BM25(k1=1.5, b=0.75)
    bm.index(corpus_tok)
    qids = list(queries.keys())
    qtok = bm25s.tokenize([queries[q] for q in qids], stopwords="en", show_progress=False)
    k = min(top_k, len(doc_ids))
    try:
        res, sc = bm.retrieve(qtok, k=k, n_threads=1, show_progress=False)
    except TypeError:
        res, sc = bm.retrieve(qtok, k=k)
    out = {}
    for i, qid in enumerate(qids):
        out[qid] = {doc_ids[res[i][j]]: float(sc[i][j]) for j in range(k)}
    return out


def _dense_run(doc_ids, doc_texts, queries, model, qpfx="", dpfx="", top_k=100, batch=32):
    if not doc_ids or not queries:
        return {qid: {} for qid in queries}
    corpus_texts = [dpfx + t for t in doc_texts] if dpfx else doc_texts
    doc_emb = model.encode(corpus_texts, batch_size=batch, show_progress_bar=False,
                           normalize_embeddings=True, convert_to_numpy=True).astype("float32")
    qids = list(queries.keys())
    qtxt = [(qpfx + queries[q]) if qpfx else queries[q] for q in qids]
    q_emb = model.encode(qtxt, batch_size=64, normalize_embeddings=True,
                         convert_to_numpy=True).astype("float32")
    if doc_emb.ndim != 2 or q_emb.ndim != 2:
        raise RuntimeError(
            f"Dense encoder produced non-2D embeddings (doc={doc_emb.shape}, "
            f"q={q_emb.shape}); check that doc_texts and queries are non-empty.")
    scores = q_emb @ doc_emb.T
    n_docs = len(doc_ids)
    k = min(top_k, n_docs)
    out = {}
    for i, qid in enumerate(qids):
        top = np.argpartition(scores[i], -k)[-k:]
        top = top[np.argsort(-scores[i][top])]
        out[qid] = {doc_ids[j]: float(scores[i][j]) for j in top}
    del doc_emb, q_emb, scores; gc.collect()
    return out


def _splade_run(doc_ids, doc_texts, queries, tok, mod, top_k=100, bs=8, max_len=128):
    if not doc_ids or not queries:
        return {qid: {} for qid in queries}
    def _enc(texts):
        rows = []
        for i in range(0, len(texts), bs):
            inp = tok(texts[i:i + bs], return_tensors="pt", padding=True,
                      truncation=True, max_length=max_len)
            inp = {k: v.to(device) for k, v in inp.items()}
            with torch.no_grad():
                vecs = torch.log(1 + torch.relu(mod(**inp).logits))
            vecs = (vecs * inp["attention_mask"].unsqueeze(-1)).max(dim=1).values
            rows.append(sp.csr_matrix(vecs.cpu().numpy()))
        return sp.vstack(rows)
    doc_vecs = _enc(doc_texts)
    qids = list(queries.keys())
    q_vecs = _enc([queries[q] for q in qids])
    scores = (q_vecs @ doc_vecs.T).toarray()
    n_docs = len(doc_ids)
    k = min(top_k, n_docs)
    out = {}
    for i, qid in enumerate(qids):
        if k >= n_docs:
            top = np.argsort(-scores[i])
        else:
            top = np.argpartition(-scores[i], k)[:k]
            top = top[np.argsort(-scores[i][top])]
        out[qid] = {doc_ids[j]: float(scores[i][j]) for j in top}
    del doc_vecs, q_vecs, scores; gc.collect()
    return out


def _medcpt_run(doc_ids, doc_texts, queries, qry_tok, qry_mod, art_tok, art_mod, top_k=100, bs=16):
    if not doc_ids or not queries:
        return {qid: {} for qid in queries}
    def _enc(texts, tok, mod, max_len):
        embs = []
        for i in range(0, len(texts), bs):
            inp = tok(texts[i:i + bs], return_tensors="pt", padding=True,
                      truncation=True, max_length=max_len)
            inp = {k: v.to(device) for k, v in inp.items()}
            with torch.no_grad():
                e = F.normalize(mod(**inp).last_hidden_state[:, 0, :], dim=-1)
            embs.append(e.cpu().numpy())
        return np.concatenate(embs, axis=0).astype("float32")
    doc_emb = _enc(doc_texts, art_tok, art_mod, 256)
    qids = list(queries.keys())
    q_emb = _enc([queries[q] for q in qids], qry_tok, qry_mod, 64)
    scores = q_emb @ doc_emb.T
    n_docs = len(doc_ids)
    k = min(top_k, n_docs)
    out = {}
    for i, qid in enumerate(qids):
        top = np.argpartition(scores[i], -k)[-k:]
        top = top[np.argsort(-scores[i][top])]
        out[qid] = {doc_ids[j]: float(scores[i][j]) for j in top}
    del doc_emb, q_emb, scores; gc.collect()
    return out


# Lazy model loaders -------------------------------------------------------
_MODELS = {}

def get_minilm():
    if "minilm" not in _MODELS:
        from sentence_transformers import SentenceTransformer
        _MODELS["minilm"] = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)
    return _MODELS["minilm"]

def get_bge():
    if "bge" not in _MODELS:
        from sentence_transformers import SentenceTransformer
        _MODELS["bge"] = SentenceTransformer("BAAI/bge-small-en-v1.5", device=device)
    return _MODELS["bge"]

def get_e5():
    if "e5" not in _MODELS:
        from sentence_transformers import SentenceTransformer
        _MODELS["e5"] = SentenceTransformer("intfloat/e5-small-v2", device=device)
    return _MODELS["e5"]

def get_splade():
    if "splade" not in _MODELS:
        from transformers import AutoTokenizer, AutoModelForMaskedLM
        name = "naver/splade-cocondenser-ensembledistil"
        tok = AutoTokenizer.from_pretrained(name)
        mod = AutoModelForMaskedLM.from_pretrained(name).to(device).eval()
        _MODELS["splade"] = (tok, mod)
    return _MODELS["splade"]

def get_medcpt():
    if "medcpt" not in _MODELS:
        from transformers import AutoTokenizer, AutoModel
        qt = AutoTokenizer.from_pretrained("ncbi/MedCPT-Query-Encoder")
        qm = AutoModel.from_pretrained("ncbi/MedCPT-Query-Encoder").to(device).eval()
        at = AutoTokenizer.from_pretrained("ncbi/MedCPT-Article-Encoder")
        am = AutoModel.from_pretrained("ncbi/MedCPT-Article-Encoder").to(device).eval()
        _MODELS["medcpt"] = (qt, qm, at, am)
    return _MODELS["medcpt"]

print("Phase-B helpers ready.")
""")


# ---------------------------------------------------------------------------
# 5. Phase A header
# ---------------------------------------------------------------------------
md("""---
## Phase A -- Bootstrap canonical seed tables

These rows come from the previous validated runs (see `final.ipynb` and
the original `next_stage` artefacts) and feed directly into Tables 3, 4
and the `n_required` table in `main.tex`. We write them once into
`OUT/tables/` so subsequent cells can update individual rows in place.

The cell is **idempotent**: existing files are not overwritten unless you
delete them first.
""")


# ---------------------------------------------------------------------------
# 6. Bootstrap seed tables
# ---------------------------------------------------------------------------
code("""# Phase A: bootstrap seed CSVs (only writes if files do not already exist).

def _write_if_missing(path, df):
    if path.exists():
        print(f"  keep {path.name}  ({len(pd.read_csv(path))} rows)")
        return
    df.to_csv(path, index=False)
    print(f"  wrote {path.name}  ({len(df)} rows)")


_canonical = pd.DataFrame([
    {"Method": "BM25",      "nDCG@10": 0.2940},
    {"Method": "Dense",     "nDCG@10": 0.3173},
    {"Method": "BGE-small", "nDCG@10": 0.3388},
    {"Method": "E5-small",  "nDCG@10": 0.3267},
    {"Method": "SPLADE",    "nDCG@10": 0.3327},
    {"Method": "MedCPT",    "nDCG@10": 0.3246},
])
_write_if_missing(OUT / "tables" / "nfcorpus_canonical.csv", _canonical)


_multi = pd.DataFrame([
    {"Dataset": "NFCorpus",      "Domain": "biomedical",   "# Docs": 3633,   "# Queries": 323,
     "BM25 nDCG@10": 0.2940, "Dense nDCG@10": 0.3173, "BGE-small nDCG@10": 0.3388,
     "E5-small nDCG@10": 0.3267, "SPLADE nDCG@10": 0.3327, "MedCPT nDCG@10": 0.3246,
     "Notes": "Canonical NFCorpus test run"},
    {"Dataset": "BioASQ-subset", "Domain": "biomedical",   "# Docs": 50000,  "# Queries": 500,
     "BM25 nDCG@10": np.nan, "Dense nDCG@10": np.nan, "BGE-small nDCG@10": np.nan,
     "E5-small nDCG@10": np.nan, "SPLADE nDCG@10": np.nan, "MedCPT nDCG@10": np.nan,
     "Notes": "Subset protocol: qrel-union plus 10k distractors"},
    {"Dataset": "TREC-COVID",    "Domain": "biomedical",   "# Docs": 171332, "# Queries": 50,
     "BM25 nDCG@10": 0.5759, "Dense nDCG@10": 0.4539, "BGE-small nDCG@10": 0.6448,
     "E5-small nDCG@10": 0.7238, "SPLADE nDCG@10": np.nan, "MedCPT nDCG@10": np.nan,
     "Notes": "SPLADE and MedCPT skipped due compute budget"},
    {"Dataset": "SciFact",       "Domain": "scientific",   "# Docs": 5183,   "# Queries": 300,
     "BM25 nDCG@10": 0.6617, "Dense nDCG@10": 0.6451, "BGE-small nDCG@10": 0.7200,
     "E5-small nDCG@10": 0.6875, "SPLADE nDCG@10": 0.6327, "MedCPT nDCG@10": 0.7103,
     "Notes": "From next_stage run"},
    {"Dataset": "ArguAna",       "Domain": "argumentation","# Docs": 8674,   "# Queries": 1406,
     "BM25 nDCG@10": 0.3606, "Dense nDCG@10": 0.3698, "BGE-small nDCG@10": 0.4287,
     "E5-small nDCG@10": 0.3100, "SPLADE nDCG@10": 0.3942, "MedCPT nDCG@10": 0.1332,
     "Notes": "From next_stage run"},
])
_write_if_missing(OUT / "tables" / "multi_dataset_ndcg10_v2.csv", _multi)


# Hoeffding n_required at delta = 0.05 for the four headline gaps.
_delta_rows = [
    ("BGE-small", "BM25",     0.3388 - 0.2940),
    ("BGE-small", "E5-small", 0.3388 - 0.3267),
    ("Hybrid",    "Dense",    0.33804189953671526 - 0.3172660881458678),
    ("Dense",     "BM25",     0.3172660881458678 - 0.2939534468857542),
]
_records = []
for a, b, d in _delta_rows:
    n_req = math.inf if abs(d) < 1e-12 else math.ceil(math.log(2 / 0.05) / (2 * d ** 2))
    _records.append({"A": a, "B": b, "observed_gap": round(d, 4),
                     "hoeffding_n_required_95pct": n_req})
_write_if_missing(OUT / "tables" / "n_required.csv", pd.DataFrame(_records))

print("\\nPhase A complete.")
""")


# ---------------------------------------------------------------------------
# 7. Phase B.1 header
# ---------------------------------------------------------------------------
md("""---
## Phase B.1 -- BioASQ retrieval pipeline

BEIR's UKP-DARMSTADT mirror does **not** redistribute the BioASQ corpus
(licence reasons; the .zip there is an HTML placeholder, which would
trigger `BadZipFile`). We use a multi-source loader instead:

1. Local BEIR-format directory at `<DATASETS_DIR>/bioasq/` (only if you
   have already downloaded BioASQ from the official challenge site).
2. HuggingFace `BeIR/bioasq` + `BeIR/bioasq-qrels`.
3. HuggingFace `enelpol/rag-mini-bioasq` (then `rag-datasets/rag-mini-bioasq`).

The result of each loader is normalised into BEIR's dict-of-dicts format
so the subset construction is identical regardless of source.
""")


# ---------------------------------------------------------------------------
# 8. BioASQ loader + subset
# ---------------------------------------------------------------------------
code("""# B.1.a: load BioASQ and build the deterministic ~50K-doc subset.
import ast as _ast

BIOASQ_DIR = DATASETS_DIR / "bioasq"
SEED = 42
TARGET_SIZE = 50_000


def _load_bioasq_local():
    if not (BIOASQ_DIR / "corpus.jsonl").exists():
        return None
    print(f"  using local BEIR-format BioASQ at {BIOASQ_DIR}")
    return GenericDataLoader(str(BIOASQ_DIR)).load(split="test")


def _load_bioasq_hf_beir():
    from datasets import load_dataset
    print("  trying HuggingFace BeIR/bioasq mirror ...", flush=True)
    corpus_ds  = load_dataset("BeIR/bioasq", "corpus",  split="corpus")
    queries_ds = load_dataset("BeIR/bioasq", "queries", split="queries")
    qrels_ds   = load_dataset("BeIR/bioasq-qrels", split="test")

    corpus = {}
    for r in corpus_ds:
        corpus[str(r["_id"])] = {"title": r.get("title", "") or "",
                                 "text":  r.get("text",  "") or ""}
    queries = {str(r["_id"]): r["text"] for r in queries_ds}
    qrels = {}
    for r in qrels_ds:
        qid, did, score = str(r["query-id"]), str(r["corpus-id"]), int(r["score"])
        qrels.setdefault(qid, {})[did] = score
    return corpus, queries, qrels


def _load_bioasq_mini():
    \"\"\"Fallback: small public BioASQ mirror. Tries `enelpol/rag-mini-bioasq`
    first (cleaned: dedup'd passages, no NaNs, `relevant_passage_ids` is a
    proper sequence-of-int) and falls back to `rag-datasets/rag-mini-bioasq`
    (original mirror, where `relevant_passage_ids` is a JSON string).\"\"\"
    from datasets import load_dataset
    last_exc = None
    for repo in ("enelpol/rag-mini-bioasq", "rag-datasets/rag-mini-bioasq"):
        try:
            print(f"  falling back to {repo} ...", flush=True)
            corpus_ds = load_dataset(repo, "text-corpus", split="passages")
            try:
                qa_ds = load_dataset(repo, "question-answer-passages", split="test")
            except Exception:
                qa_ds = load_dataset(repo, "question-answer-passages", split="train")
        except Exception as exc:
            last_exc = exc
            continue

        corpus = {}
        for r in corpus_ds:
            did  = str(r.get("id"))
            text = r.get("passage", "") or ""
            if not text or text == "nan":
                continue
            corpus[did] = {"title": "", "text": text}
        if not corpus:
            continue

        queries, qrels = {}, {}
        for i, r in enumerate(qa_ds):
            qid = str(r.get("id", i))
            q   = r.get("question", "") or ""
            rel = r.get("relevant_passage_ids", []) or []
            if isinstance(rel, str):
                try:
                    rel = json.loads(rel)
                except (ValueError, TypeError):
                    try:
                        rel = _ast.literal_eval(rel)
                    except (ValueError, SyntaxError):
                        rel = []
            rel_ids = [str(d) for d in rel if str(d) in corpus]
            if not rel_ids or not q:
                continue
            queries[qid] = q
            qrels[qid] = {d: 1 for d in rel_ids}

        if not queries:
            continue
        return corpus, queries, qrels

    if last_exc is not None:
        raise last_exc
    raise RuntimeError("rag-mini-bioasq mirrors loaded but produced no QAP triples")


_corpus_full = _queries_full = _qrels_full = None
BIOASQ_SOURCE = None
for _label, _fn in [
    ("local BEIR directory",      _load_bioasq_local),
    ("HuggingFace BeIR/bioasq",   _load_bioasq_hf_beir),
    ("rag-mini-bioasq fallback",  _load_bioasq_mini),
]:
    try:
        _result = _fn()
    except Exception as exc:
        print(f"  [{_label}] failed: {type(exc).__name__}: {exc}")
        continue
    if _result is None:
        continue
    _corpus_full, _queries_full, _qrels_full = _result
    BIOASQ_SOURCE = _label
    break

if _corpus_full is None:
    raise RuntimeError(
        "All BioASQ loaders failed. To use the official corpus, register at "
        "http://bioasq.org/, download the BEIR-format release, and unpack "
        f"it into {BIOASQ_DIR} so it contains corpus.jsonl, queries.jsonl, "
        "and qrels/test.tsv.")

print(f"  source: {BIOASQ_SOURCE}")
print(f"  full corpus: {len(_corpus_full):,} docs  |  "
      f"{len(_queries_full):,} queries  |  qrels: {len(_qrels_full):,}")

TARGET_SIZE = min(TARGET_SIZE, len(_corpus_full))

rng = np.random.default_rng(SEED)
qrel_union = set()
for qid, rels in _qrels_full.items():
    qrel_union.update(rels.keys())
qrel_union &= set(_corpus_full.keys())
print(f"  qrel-union size: {len(qrel_union):,}")

remaining = list(set(_corpus_full.keys()) - qrel_union)
n_distractors = max(0, min(TARGET_SIZE - len(qrel_union), 10_000, len(remaining)))
distractor_ids = (rng.choice(remaining, size=n_distractors, replace=False).tolist()
                  if n_distractors > 0 else [])

subset_ids = sorted(qrel_union | set(distractor_ids))
print(f"  subset size: {len(subset_ids):,} "
      f"({len(qrel_union):,} qrel-union + {n_distractors:,} distractors)")

_manifest_path = OUT / "datasets" / "bioasq_subset_doc_ids.json"
with open(_manifest_path, "w") as fh:
    json.dump({"seed": SEED, "source": BIOASQ_SOURCE,
               "protocol": "union_of_qrel_relevant_docs_plus_distractors",
               "target_size": TARGET_SIZE, "n_qrel_union": len(qrel_union),
               "n_distractors": n_distractors, "n_total": len(subset_ids),
               "doc_ids": subset_ids}, fh)
print(f"  manifest -> {_manifest_path}")

assert len(subset_ids) > 0, (
    f"BioASQ subset is empty (source={BIOASQ_SOURCE!r}). "
    "Inspect the loader chain above.")
assert len(_qrels_full) > 0, (
    f"No qrels parsed from source={BIOASQ_SOURCE!r}; downstream evaluation "
    "would be vacuous.")
""")


# ---------------------------------------------------------------------------
# 9. BioASQ subset corpus + sample
# ---------------------------------------------------------------------------
code("""# B.1.b: build the subset corpus and a 500-query deterministic sample.
subset_set = set(subset_ids)
bioasq_corpus = {d: _corpus_full[d] for d in subset_ids}

bioasq_qrels = {}
for qid, rels in _qrels_full.items():
    inner = {d: r for d, r in rels.items() if d in subset_set}
    if inner:
        bioasq_qrels[qid] = inner

eligible_qids = sorted(bioasq_qrels.keys())
rng = np.random.default_rng(SEED)
sample_size = min(500, len(eligible_qids))
sampled = sorted(rng.choice(eligible_qids, size=sample_size, replace=False).tolist())
bioasq_queries = {qid: _queries_full[qid] for qid in sampled if qid in _queries_full}
bioasq_qrels   = {qid: bioasq_qrels[qid]  for qid in bioasq_queries}
print(f"  sampled {len(bioasq_queries)} queries | qrel-coverage in subset OK")

doc_ids_b, doc_texts_b = _prep_corpus(bioasq_corpus)
print(f"  prepared {len(doc_ids_b):,} documents for retrieval")

assert len(bioasq_queries) > 0, "BioASQ query sample is empty (rerun B.1.a)."
assert len(doc_ids_b)      > 0, "BioASQ subset corpus is empty (rerun B.1.a)."
""")


# ---------------------------------------------------------------------------
# 10. BioASQ retrievers
# ---------------------------------------------------------------------------
code("""# B.1.c: run all 6 retrievers on the BioASQ subset (with disk caching).
BIOASQ_CACHE = CACHE_DIR / "bioasq_results.pkl"
EXPECTED_RETRIEVERS = ["BM25", "Dense", "BGE-small", "E5-small", "SPLADE", "MedCPT"]


def _cache_complete(blob, query_ids):
    if not isinstance(blob, dict):
        return False
    if set(blob.keys()) != set(EXPECTED_RETRIEVERS):
        return False
    qid_set = set(query_ids)
    for name, res in blob.items():
        if not isinstance(res, dict) or set(res.keys()) != qid_set:
            return False
        if not any(res[qid] for qid in qid_set):
            return False
    return True


bioasq_results = None
if BIOASQ_CACHE.exists():
    try:
        with open(BIOASQ_CACHE, "rb") as fh:
            _cached = pickle.load(fh)
        if _cache_complete(_cached, bioasq_queries.keys()):
            bioasq_results = _cached
            print("Loaded BioASQ retrieval results from cache.")
        else:
            print("BioASQ cache is incomplete or empty; re-running retrievers.")
    except Exception as exc:
        print(f"BioASQ cache unreadable ({exc}); re-running retrievers.")

if bioasq_results is None:
    bioasq_results = {}
    print("BM25 ...", flush=True)
    bioasq_results["BM25"]      = _bm25_run(doc_ids_b, doc_texts_b, bioasq_queries, top_k=100)
    print("Dense (MiniLM) ...", flush=True)
    bioasq_results["Dense"]     = _dense_run(doc_ids_b, doc_texts_b, bioasq_queries, get_minilm(),  top_k=100)
    print("BGE-small ...", flush=True)
    bioasq_results["BGE-small"] = _dense_run(doc_ids_b, doc_texts_b, bioasq_queries, get_bge(),     top_k=100)
    print("E5-small ...", flush=True)
    bioasq_results["E5-small"]  = _dense_run(doc_ids_b, doc_texts_b, bioasq_queries, get_e5(),
                                              qpfx="query: ", dpfx="passage: ", top_k=100)
    print("SPLADE ...", flush=True)
    _splade_tok, _splade_mod = get_splade()
    bioasq_results["SPLADE"]    = _splade_run(doc_ids_b, doc_texts_b, bioasq_queries,
                                               _splade_tok, _splade_mod, top_k=100)
    print("MedCPT ...", flush=True)
    _qt, _qm, _at, _am = get_medcpt()
    bioasq_results["MedCPT"]    = _medcpt_run(doc_ids_b, doc_texts_b, bioasq_queries,
                                               _qt, _qm, _at, _am, top_k=100)
    with open(BIOASQ_CACHE, "wb") as fh:
        pickle.dump(bioasq_results, fh)

bioasq_eval = {name: evaluate_retriever(res, bioasq_qrels) for name, res in bioasq_results.items()}
for name, ev in bioasq_eval.items():
    print(f"  {name:<10} nDCG@10 = {ev['aggregate']['nDCG@10']:.4f}")
""")


# ---------------------------------------------------------------------------
# 11. BioASQ backfill + bootstrap
# ---------------------------------------------------------------------------
code("""# B.1.d: backfill multi_dataset_ndcg10_v2.csv and emit paired bootstrap CI.
md_path = OUT / "tables" / "multi_dataset_ndcg10_v2.csv"
md = pd.read_csv(md_path)
mask = md["Dataset"] == "BioASQ-subset"
for col, name in [
    ("BM25 nDCG@10",      "BM25"),
    ("Dense nDCG@10",     "Dense"),
    ("BGE-small nDCG@10", "BGE-small"),
    ("E5-small nDCG@10",  "E5-small"),
    ("SPLADE nDCG@10",    "SPLADE"),
    ("MedCPT nDCG@10",    "MedCPT"),
]:
    md.loc[mask, col] = round(bioasq_eval[name]["aggregate"]["nDCG@10"], 4)
md.loc[mask, "# Docs"]    = len(doc_ids_b)
md.loc[mask, "# Queries"] = len(bioasq_queries)
md.loc[mask, "Notes"] = (f"Subset of qrel-union ({len(qrel_union):,}) + "
                         f"{n_distractors:,} distractors, seed=42, "
                         f"{len(bioasq_queries)}-query sample")
md.to_csv(md_path, index=False)
print(f"Updated {md_path}")
print(md)


# Per-query nDCG@10 -> parquet (used later by the router)
qids = sorted(bioasq_queries.keys())
perquery = {"qid": qids}
for name, ev in bioasq_eval.items():
    perquery[name] = [ev["per_query"][q]["nDCG@10"] for q in qids]
pq_df = pd.DataFrame(perquery)
pq_df.to_parquet(CACHE_DIR / "bioasq_perquery.parquet", index=False)
print(f"Per-query nDCG@10 -> {CACHE_DIR / 'bioasq_perquery.parquet'}")


def paired_bootstrap_ci(a, b, B=10_000, alpha=0.05, seed=42):
    a, b = np.asarray(a, float), np.asarray(b, float)
    diffs = a - b
    rng = np.random.default_rng(seed)
    n = len(diffs)
    boot = np.array([diffs[rng.integers(0, n, n)].mean() for _ in range(B)])
    lo, hi = np.quantile(boot, [alpha / 2, 1 - alpha / 2])
    return float(diffs.mean()), float(lo), float(hi), float((boot > 0).mean())


m, lo, hi, p_pos = paired_bootstrap_ci(pq_df["MedCPT"].values, pq_df["BM25"].values)
print(f"\\nMedCPT - BM25 (BioASQ): mean={m:+.4f} CI95=[{lo:+.4f}, {hi:+.4f}] P(MedCPT>BM25)={p_pos:.3f}")
pd.DataFrame([{
    "contrast":   "MedCPT - BM25",
    "dataset":    "BioASQ-subset",
    "n_queries":  len(qids),
    "mean_diff":  round(m, 4),
    "ci_lo":      round(lo, 4),
    "ci_hi":      round(hi, 4),
    "p_a_gt_b":   round(p_pos, 3),
}]).to_csv(OUT / "tables" / "bioasq_paired_bootstrap.csv", index=False)
print(f"Saved bootstrap CI -> {OUT / 'tables' / 'bioasq_paired_bootstrap.csv'}")
""")


# ---------------------------------------------------------------------------
# 12. Phase B.2 header
# ---------------------------------------------------------------------------
md("""---
## Phase B.2 -- BGE cross-encoder reranking (+ optional MedCPT-CE)

Adds a stronger reranking comparison on top of the strongest first-stage
retriever for each dataset (Table 5 in `main.tex`). The reranker is
`BAAI/bge-reranker-base`, a cross-encoder shipped with the BGE family
that scores `(query, passage)` pairs end-to-end -- it is the simplest
"stronger reranker" the instructor asked for and avoids the late-
interaction tooling (pylate / ColBERT) entirely:

**VRAM (Colab T4):** the cross-encoder loads in **fp16** on CUDA and uses
`RERANK_BATCH=8` by default. If you still hit OOM, set
`os.environ["RAGEVAL_RERANK_BATCH"]="4"` before the import cell, or restart
the runtime and skip MedCPT-CE (comment out the next optional cell).

* NFCorpus: BGE-small@100 -> BGE-reranker; BM25@100 -> BGE-reranker.
* BioASQ-subset: BM25@100 -> BGE-reranker.
* TREC-COVID: E5-small@100 -> BGE-reranker.
* SciFact: BGE-small@100 -> BGE-reranker.
* ArguAna: BGE-small@100 -> BGE-reranker.

Optionally adds the MedCPT-Cross-Encoder on the two biomedical datasets
for an in-domain comparison.
""")


# ---------------------------------------------------------------------------
# 13. Load BGE cross-encoder reranker
# ---------------------------------------------------------------------------
code("""# B.2.a: load the BGE cross-encoder reranker via sentence-transformers.
from sentence_transformers import CrossEncoder

RERANKER_MODEL_NAME = "BAAI/bge-reranker-base"
print(f"Loading {RERANKER_MODEL_NAME} ...")
_ce_kw = {}
if device == "cuda":
    # fp16 halves VRAM vs fp32; essential on Colab T4 when a bi-encoder is also loaded.
    _ce_kw["model_kwargs"] = {"torch_dtype": torch.float16}
reranker_model = CrossEncoder(RERANKER_MODEL_NAME, device=device, max_length=512, **_ce_kw)
print("BGE reranker ready (cross-encoder, fp16 on CUDA; scores (query, passage) pairs).")
""")


# ---------------------------------------------------------------------------
# 14. Cross-encoder reranker helper
# ---------------------------------------------------------------------------
code("""# B.2.b: BGE cross-encoder reranker over precomputed top-100 candidates.

def cross_encoder_rerank(queries, corpus, candidate_ids_per_query, model, batch_size=None):
    \"\"\"Score every (query, candidate) pair with a sentence-transformers
    CrossEncoder and return a dict-of-dicts in the same shape as the
    first-stage retrievers (so it slots into the existing harness).\"\"\"
    if batch_size is None:
        batch_size = RERANK_BATCH
    out = {}
    for qid in tqdm(queries, desc="BGE-reranker rerank"):
        cand_ids = candidate_ids_per_query.get(qid, [])
        if not cand_ids:
            out[qid] = {}
            continue
        cand_texts = [(corpus[d].get("title", "") + " " + corpus[d].get("text", "")).strip()
                      for d in cand_ids]
        pairs = [(queries[qid], t) for t in cand_texts]
        scores = model.predict(pairs, batch_size=batch_size,
                                show_progress_bar=False,
                                convert_to_numpy=True)
        order = np.argsort(-np.asarray(scores, dtype=float))
        out[qid] = {cand_ids[i]: float(scores[i]) for i in order}
    return out


def first_stage_top100(name, doc_ids, doc_texts, queries, retriever):
    if retriever == "BM25":
        return _bm25_run(doc_ids, doc_texts, queries, top_k=100)
    if retriever == "BGE-small":
        return _dense_run(doc_ids, doc_texts, queries, get_bge(), top_k=100)
    if retriever == "E5-small":
        return _dense_run(doc_ids, doc_texts, queries, get_e5(),
                          qpfx="query: ", dpfx="passage: ", top_k=100)
    raise ValueError(retriever)


RERANK_CACHE = CACHE_DIR / "reranked_results.pkl"
reranked_cache = pickle.load(open(RERANK_CACHE, "rb")) if RERANK_CACHE.exists() else {}


def _rerank_entry_ok(entry, queries):
    if not isinstance(entry, dict):
        return False
    if set(entry.keys()) != set(queries.keys()):
        return False
    return any(entry[qid] for qid in queries)


def _do_dataset(ds_name, corpus, queries, qrels, first_stage_results, first_stage_name):
    key = f"{ds_name}|{first_stage_name}|bge_reranker"
    if key in reranked_cache and _rerank_entry_ok(reranked_cache[key], queries):
        print(f"  cache hit: {key}")
        return reranked_cache[key]
    if key in reranked_cache:
        print(f"  cache hit was empty/incomplete; recomputing {key}")
    cand = {qid: list(first_stage_results[qid].keys()) for qid in queries}
    rer = cross_encoder_rerank(queries, corpus, cand, reranker_model, batch_size=RERANK_BATCH)
    cuda_gc()
    reranked_cache[key] = rer
    with open(RERANK_CACHE, "wb") as fh:
        pickle.dump(reranked_cache, fh)
    return rer


print("Reranker harness ready.")
""")


# ---------------------------------------------------------------------------
# 15. Reranker driver
# ---------------------------------------------------------------------------
code("""# B.2.c: driver -- run reranking over the 5 datasets.
def load_or_build_first_stage_nfcorpus(retriever_name):
    NF_DIR = DATASETS_DIR / "nfcorpus"
    if not NF_DIR.exists():
        beir_util.download_and_unzip(
            "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/nfcorpus.zip",
            str(DATASETS_DIR))
    corpus, queries, qrels = GenericDataLoader(str(NF_DIR)).load(split="test")
    doc_ids, doc_texts = _prep_corpus(corpus)
    fs = first_stage_top100("NFCorpus", doc_ids, doc_texts, queries, retriever_name)
    return corpus, queries, qrels, fs


def _agg_ndcg10(results, qrels):
    return evaluate_retriever(results, qrels)["aggregate"]["nDCG@10"]


def _row(ds, fs_name, fs_n, sys_name, rer_n):
    return dict(Dataset=ds, FirstStage=fs_name,
                FirstStage_nDCG10=round(fs_n, 4),
                RerankedSystem=sys_name,
                Reranked_nDCG10=round(rer_n, 4),
                Delta=round(rer_n - fs_n, 4),
                Status="done")


REranker_RESULTS = []

print("\\n[NFCorpus] BGE-small first-stage -> BGE-reranker rerank")
nf_corpus, nf_queries, nf_qrels, nf_fs_bge = load_or_build_first_stage_nfcorpus("BGE-small")
nf_rer_bge = _do_dataset("NFCorpus", nf_corpus, nf_queries, nf_qrels, nf_fs_bge, "BGE-small")
nf_first_bge = _agg_ndcg10(nf_fs_bge, nf_qrels)
nf_rer_bge_n = _agg_ndcg10(nf_rer_bge, nf_qrels)
print(f"  first={nf_first_bge:.4f}  reranked={nf_rer_bge_n:.4f}  delta={nf_rer_bge_n-nf_first_bge:+.4f}")
REranker_RESULTS.append(_row("NFCorpus", "BGE-small", nf_first_bge,
                              "BGE-small+BGE-reranker@100", nf_rer_bge_n))

print("\\n[NFCorpus] BM25 first-stage -> BGE-reranker rerank")
_, _, _, nf_fs_bm = load_or_build_first_stage_nfcorpus("BM25")
nf_rer_bm = _do_dataset("NFCorpus", nf_corpus, nf_queries, nf_qrels, nf_fs_bm, "BM25")
nf_first_bm = _agg_ndcg10(nf_fs_bm, nf_qrels)
nf_rer_bm_n = _agg_ndcg10(nf_rer_bm, nf_qrels)
print(f"  first={nf_first_bm:.4f}  reranked={nf_rer_bm_n:.4f}  delta={nf_rer_bm_n-nf_first_bm:+.4f}")
REranker_RESULTS.append(_row("NFCorpus", "BM25", nf_first_bm,
                              "BM25+BGE-reranker@100", nf_rer_bm_n))

print("\\n[BioASQ-subset] BM25 first-stage -> BGE-reranker rerank")
ba_rer_bm = _do_dataset("BioASQ-subset", bioasq_corpus, bioasq_queries, bioasq_qrels,
                         bioasq_results["BM25"], "BM25")
ba_first_bm = _agg_ndcg10(bioasq_results["BM25"], bioasq_qrels)
ba_rer_bm_n = _agg_ndcg10(ba_rer_bm, bioasq_qrels)
print(f"  first={ba_first_bm:.4f}  reranked={ba_rer_bm_n:.4f}  delta={ba_rer_bm_n-ba_first_bm:+.4f}")
REranker_RESULTS.append(_row("BioASQ-subset", "BM25", ba_first_bm,
                              "BM25+BGE-reranker@100", ba_rer_bm_n))


OTHER_DS = [
    ("TREC-COVID", "trec-covid", "E5-small"),
    ("SciFact",   "scifact",   "BGE-small"),
    ("ArguAna",   "arguana",   "BGE-small"),
]
for ds_name, beir_name, first_stage_name in OTHER_DS:
    print(f"\\n[{ds_name}] {first_stage_name} first-stage -> BGE-reranker rerank")
    ds_dir = DATASETS_DIR / beir_name
    if not ds_dir.exists():
        beir_util.download_and_unzip(
            f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{beir_name}.zip",
            str(DATASETS_DIR))
    corpus, queries, qrels = GenericDataLoader(str(ds_dir)).load(split="test")
    doc_ids, doc_texts = _prep_corpus(corpus)
    fs = first_stage_top100(ds_name, doc_ids, doc_texts, queries, first_stage_name)
    cuda_gc()
    rer = _do_dataset(ds_name, corpus, queries, qrels, fs, first_stage_name)
    fs_n  = _agg_ndcg10(fs,  qrels)
    rer_n = _agg_ndcg10(rer, qrels)
    print(f"  first={fs_n:.4f}  reranked={rer_n:.4f}  delta={rer_n-fs_n:+.4f}")
    REranker_RESULTS.append(_row(ds_name, first_stage_name, fs_n,
                                  f"{first_stage_name}+BGE-reranker@100", rer_n))


rer_df = pd.DataFrame(REranker_RESULTS)
rer_df.to_csv(OUT / "tables" / "reranker_ndcg10.csv", index=False)
print(f"\\nSaved -> {OUT / 'tables' / 'reranker_ndcg10.csv'}")
print(rer_df)


def perquery_ndcg10(results, qrels):
    qids = sorted(set(results.keys()) & set(qrels.keys()))
    out = {}
    for qid in qids:
        ranking = [d for d, _ in sorted(results[qid].items(), key=lambda x: -x[1])]
        out[qid] = compute_ndcg(ranking, qrels[qid], 10)
    return out


nf_rer_bge_perq = perquery_ndcg10(nf_rer_bge, nf_qrels)
pd.DataFrame({"qid": list(nf_rer_bge_perq.keys()),
              "BGE_BGEReranker_at100": list(nf_rer_bge_perq.values())}
             ).to_parquet(CACHE_DIR / "reranked_perquery.parquet", index=False)
print(f"NFCorpus per-query reranked nDCG@10 -> {CACHE_DIR / 'reranked_perquery.parquet'}")
""")


# ---------------------------------------------------------------------------
# 16. MedCPT-CE reranker (optional)
# ---------------------------------------------------------------------------
code("""# B.2.d: MedCPT cross-encoder reranker on biomedical datasets only (optional).
try:
    from transformers import AutoTokenizer as _AT, AutoModelForSequenceClassification as _AS
    medcpt_ce_tok = _AT.from_pretrained("ncbi/MedCPT-Cross-Encoder")
    _mce_dtype = torch.float16 if device == "cuda" else torch.float32
    medcpt_ce_mod = _AS.from_pretrained(
        "ncbi/MedCPT-Cross-Encoder", torch_dtype=_mce_dtype,
    ).to(device).eval()
    HAVE_MEDCPT_CE = True
except Exception as exc:
    print(f"MedCPT cross-encoder unavailable ({exc}); skipping this sub-section.")
    HAVE_MEDCPT_CE = False


def medcpt_ce_rerank(queries, corpus, candidate_ids_per_query, batch_size=8):
    out = {}
    for qid in tqdm(queries, desc="MedCPT-CE rerank"):
        cand_ids = candidate_ids_per_query.get(qid, [])
        if not cand_ids:
            out[qid] = {}
            continue
        cand_texts = [(corpus[d].get("title", "") + " " + corpus[d].get("text", "")).strip()
                      for d in cand_ids]
        q = queries[qid]
        scores = []
        for i in range(0, len(cand_ids), batch_size):
            inp = medcpt_ce_tok([(q, t) for t in cand_texts[i:i + batch_size]],
                                 return_tensors="pt", truncation=True, padding=True,
                                 max_length=512).to(device)
            with torch.no_grad():
                lg = medcpt_ce_mod(**inp).logits.squeeze(-1).cpu().numpy()
            scores.extend(lg.tolist())
        order = np.argsort(-np.array(scores))
        out[qid] = {cand_ids[i]: float(scores[i]) for i in order}
    return out


if HAVE_MEDCPT_CE:
    medcpt_ce_rows = []

    rer = medcpt_ce_rerank(nf_queries, nf_corpus,
                            {qid: list(nf_fs_bge[qid].keys()) for qid in nf_queries})
    n_after = _agg_ndcg10(rer, nf_qrels)
    medcpt_ce_rows.append(dict(Dataset="NFCorpus", FirstStage="BGE-small",
                                FirstStage_nDCG10=round(nf_first_bge, 4),
                                RerankedSystem="BGE-small+MedCPT-CE@100",
                                Reranked_nDCG10=round(n_after, 4),
                                Delta=round(n_after - nf_first_bge, 4),
                                Status="done"))

    rer = medcpt_ce_rerank(bioasq_queries, bioasq_corpus,
                            {qid: list(bioasq_results["BM25"][qid].keys())
                             for qid in bioasq_queries})
    n_after = _agg_ndcg10(rer, bioasq_qrels)
    medcpt_ce_rows.append(dict(Dataset="BioASQ-subset", FirstStage="BM25",
                                FirstStage_nDCG10=round(ba_first_bm, 4),
                                RerankedSystem="BM25+MedCPT-CE@100",
                                Reranked_nDCG10=round(n_after, 4),
                                Delta=round(n_after - ba_first_bm, 4),
                                Status="done"))

    rerank_df = pd.read_csv(OUT / "tables" / "reranker_ndcg10.csv")
    rerank_df = pd.concat([rerank_df, pd.DataFrame(medcpt_ce_rows)], ignore_index=True)
    rerank_df.to_csv(OUT / "tables" / "reranker_ndcg10.csv", index=False)
    print(rerank_df)
""")


# ---------------------------------------------------------------------------
# 17. Phase B.3 header
# ---------------------------------------------------------------------------
md("""---
## Phase B.3 -- Per-query router

Trains a logistic + LightGBM router that, given six per-query features,
chooses among `{BM25, BGE-small, Hybrid(alpha=0.40), BGE-small+BGE-reranker}`.
Reports `Always_*` baselines and the oracle upper bound for context.

Outputs:

* `router_test.csv` -- Table 6
* `router_feature_importance.csv` -- narrative
* `router_train_test_gap.csv` -- Table 7
""")


# ---------------------------------------------------------------------------
# 18. Router features
# ---------------------------------------------------------------------------
code("""# B.3.a: build per-query features (gap, len, tech, querytype, medvocab, bm25top1).
NF_DIR = DATASETS_DIR / "nfcorpus"
if not NF_DIR.exists():
    beir_util.download_and_unzip(
        "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/nfcorpus.zip",
        str(DATASETS_DIR))

splits = {}
for split in ["train", "dev", "test"]:
    corpus, queries, qrels = GenericDataLoader(str(NF_DIR)).load(split=split)
    splits[split] = {"corpus": corpus, "queries": queries, "qrels": qrels}

nf_doc_ids, nf_doc_texts = _prep_corpus(splits["test"]["corpus"])
nf_doc_text_by_id = dict(zip(nf_doc_ids, nf_doc_texts))

WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9\\-]+")
def _tokens(s):
    return [t.lower() for t in WORD_RE.findall(s)]

import collections
df_count = collections.Counter()
for txt in nf_doc_texts:
    df_count.update(set(_tokens(txt)))
N = len(nf_doc_ids)
idf = {t: math.log((N + 1) / (df_count[t] + 1)) + 1.0 for t in df_count}

MED_AFFIX = ("mab", "tinib", "vir", "olol", "azepam", "azole", "statin", "cycline",
             "itis", "osis", "emia", "oma", "pathy", "cardio", "neuro", "hepato",
             "renal", "gastro", "pulmonary")

def _med_share(toks):
    if not toks:
        return 0.0
    return sum(any(t.endswith(x) or x in t for x in MED_AFFIX) for t in toks) / len(toks)


def _gap(qtxt, rels):
    qset = set(_tokens(qtxt))
    if not qset:
        return 0.0
    best = 0.0
    for d in rels:
        if d not in nf_doc_text_by_id:
            continue
        dset = set(_tokens(nf_doc_text_by_id[d]))
        ov = len(qset & dset) / len(qset)
        best = max(best, ov)
    return 1.0 - best


import bm25s
_bm25_corpus_tok = bm25s.tokenize(nf_doc_texts, stopwords="en", show_progress=False)
_bm25_full = bm25s.BM25(k1=1.5, b=0.75)
_bm25_full.index(_bm25_corpus_tok)

def _bm25_top1_score(q):
    q_tok = bm25s.tokenize([q], stopwords="en", show_progress=False)
    try:
        _, sc = _bm25_full.retrieve(q_tok, k=1, n_threads=1, show_progress=False)
    except TypeError:
        _, sc = _bm25_full.retrieve(q_tok, k=1)
    if len(sc) == 0 or len(sc[0]) == 0:
        return 0.0
    return float(sc[0][0])


QUESTION_RE = re.compile(r"^\\s*(what|who|how|why|when|where|is|are|does|do|can|should)\\b", re.I)


feature_rows = []
for split, d in splits.items():
    for qid, qtxt in d["queries"].items():
        toks = _tokens(qtxt)
        rels = d["qrels"].get(qid, {})
        feature_rows.append({
            "split": split, "qid": qid,
            "gap":         _gap(qtxt, rels),
            "len":         len(toks),
            "tech":        float(np.mean([idf.get(t, 1.0) for t in toks])) if toks else 0.0,
            "is_question": int(bool(QUESTION_RE.match(qtxt))),
            "med_share":   _med_share(toks),
            "bm25_top1":   _bm25_top1_score(qtxt),
        })
features_df = pd.DataFrame(feature_rows)
print("features:", features_df.shape)
features_df.head()
""")


# ---------------------------------------------------------------------------
# 19. Per-query nDCG@10 for 4 strategies
# ---------------------------------------------------------------------------
code("""# B.3.b: per-query nDCG@10 for each strategy on every NFCorpus query.
ALPHA_HYBRID = 0.40

def _per_q_ndcg10(results, qrels):
    out = {}
    for qid, doc_scores in results.items():
        if qid not in qrels:
            continue
        ranking = [d for d, _ in sorted(doc_scores.items(), key=lambda x: -x[1])]
        out[qid] = compute_ndcg(ranking, qrels[qid], 10)
    return out


ROUTER_CACHE = CACHE_DIR / "router_strategy_perq.parquet"
_router_cache_ok = False
if ROUTER_CACHE.exists():
    try:
        _candidate = pd.read_parquet(ROUTER_CACHE)
        _need_cols = {"split", "qid", "BM25", "BGE-small", "Hybrid", "BGE+BGE-reranker"}
        _need_splits = {"train", "dev", "test"}
        if (_need_cols.issubset(_candidate.columns) and
                _need_splits.issubset(set(_candidate["split"].unique()))):
            perq_strategies = _candidate
            _router_cache_ok = True
            print("Loaded per-query strategy nDCG@10 cache.")
        else:
            print("Router cache missing splits/columns; rebuilding.")
    except Exception as exc:
        print(f"Router cache unreadable ({exc}); rebuilding.")

if not _router_cache_ok:
    bge = get_bge()
    rows = []
    for split, d in splits.items():
        bm25_res = _bm25_run(nf_doc_ids, nf_doc_texts, d["queries"], top_k=100)
        bge_res  = _dense_run(nf_doc_ids, nf_doc_texts, d["queries"], bge, top_k=100)

        def _mm(a):
            lo, hi = a.min(), a.max()
            return (a - lo) / (hi - lo) if hi > lo else np.zeros_like(a)

        hyb_res = {}
        for qid in d["queries"]:
            sb = bm25_res.get(qid, {})
            sd = bge_res.get(qid, {})
            cand = sorted(set(sb) | set(sd))
            if not cand:
                hyb_res[qid] = {}
                continue
            sb_arr = np.array([sb.get(c, 0.0) for c in cand])
            sd_arr = np.array([sd.get(c, 0.0) for c in cand])
            hyb = ALPHA_HYBRID * _mm(sb_arr) + (1.0 - ALPHA_HYBRID) * _mm(sd_arr)
            hyb_res[qid] = {c: float(s) for c, s in zip(cand, hyb)}

        cand = {qid: list(bge_res[qid].keys()) for qid in d["queries"]}
        rer = cross_encoder_rerank(d["queries"], d["corpus"], cand, reranker_model,
                                    batch_size=RERANK_BATCH)

        bm25_q = _per_q_ndcg10(bm25_res, d["qrels"])
        bge_q  = _per_q_ndcg10(bge_res,  d["qrels"])
        hyb_q  = _per_q_ndcg10(hyb_res,  d["qrels"])
        rer_q  = _per_q_ndcg10(rer,      d["qrels"])
        for qid in d["queries"]:
            rows.append({"split": split, "qid": qid,
                          "BM25":             bm25_q.get(qid, 0.0),
                          "BGE-small":        bge_q.get(qid, 0.0),
                          "Hybrid":           hyb_q.get(qid, 0.0),
                          "BGE+BGE-reranker": rer_q.get(qid, 0.0)})
    perq_strategies = pd.DataFrame(rows)
    perq_strategies.to_parquet(ROUTER_CACHE, index=False)

print(perq_strategies.groupby("split")[["BM25", "BGE-small", "Hybrid", "BGE+BGE-reranker"]].mean().round(4))
""")


# ---------------------------------------------------------------------------
# 20. Train router
# ---------------------------------------------------------------------------
code("""# B.3.c: train logistic + LightGBM router; report Always-baselines and oracle.
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import lightgbm as lgb

STRATS = ["BM25", "BGE-small", "Hybrid", "BGE+BGE-reranker"]

df_all = features_df.merge(perq_strategies, on=["split", "qid"], how="inner")
labels = df_all[STRATS].values.argmax(axis=1)
df_all["label"] = labels
print("strategy distribution (label = argmax over 4 strategies):")
print(pd.Series(labels).value_counts().rename(lambda i: STRATS[i]))

FEATURES = ["gap", "len", "tech", "is_question", "med_share", "bm25_top1"]

train_mask = df_all["split"].isin(["train", "dev"])
test_mask  = df_all["split"] == "test"

Xtr = df_all.loc[train_mask, FEATURES].values
ytr = df_all.loc[train_mask, "label"].values
Xte = df_all.loc[test_mask,  FEATURES].values
yte = df_all.loc[test_mask,  "label"].values

log_pipe = make_pipeline(StandardScaler(),
                          LogisticRegression(max_iter=2000))
log_pipe.fit(Xtr, ytr)

gbm = lgb.LGBMClassifier(num_leaves=31, learning_rate=0.05, n_estimators=300,
                          objective="multiclass", num_class=4, random_state=42)
gbm.fit(Xtr, ytr)


def _route(model, X, df_split):
    preds = model.predict(X)
    chosen = df_split[STRATS].values[np.arange(len(df_split)), preds]
    return float(np.mean(chosen)), preds


test_df = df_all.loc[test_mask].reset_index(drop=True)

log_test, log_preds_te = _route(log_pipe, Xte, test_df)
gbm_test, gbm_preds_te = _route(gbm,      Xte, test_df)
oracle      = float(np.mean(test_df[STRATS].values.max(axis=1)))
always_bge  = float(np.mean(test_df["BGE-small"].values))
always_hyb  = float(np.mean(test_df["Hybrid"].values))
always_rer  = float(np.mean(test_df["BGE+BGE-reranker"].values))
always_bm   = float(np.mean(test_df["BM25"].values))

router_rows = [
    {"Method": "Always_BM25",                    "nDCG@10": round(always_bm,  4), "Status": "baseline"},
    {"Method": "Always_BGE_small",               "nDCG@10": round(always_bge, 4), "Status": "baseline"},
    {"Method": "Static_Hybrid_alpha_0.40",       "nDCG@10": round(always_hyb, 4), "Status": "baseline"},
    {"Method": "Always_BGE_small+BGE_reranker",  "nDCG@10": round(always_rer, 4), "Status": "baseline"},
    {"Method": "Logistic_router",                "nDCG@10": round(log_test,   4), "Status": "learned"},
    {"Method": "LightGBM_router",                "nDCG@10": round(gbm_test,   4), "Status": "learned"},
    {"Method": "Oracle_router (upper bound)",    "nDCG@10": round(oracle,     4), "Status": "oracle"},
]
router_df = pd.DataFrame(router_rows)
router_df.to_csv(OUT / "tables" / "router_test.csv", index=False)
print(router_df)


fi = pd.DataFrame({"feature": FEATURES,
                    "gain": gbm.booster_.feature_importance(importance_type="gain")})
fi.sort_values("gain", ascending=False, inplace=True)
fi.to_csv(OUT / "tables" / "router_feature_importance.csv", index=False)
print("\\nFeature importance (gain):")
print(fi)
""")


# ---------------------------------------------------------------------------
# 21. Train/test gap
# ---------------------------------------------------------------------------
code("""# B.3.d: router train/test generalisation gap (Section 4.10 anchor).
train_df = df_all.loc[train_mask].reset_index(drop=True)
log_train, _ = _route(log_pipe, Xtr, train_df)
gbm_train, _ = _route(gbm,      Xtr, train_df)


def _risk(mean_ndcg10):
    return 1.0 - mean_ndcg10


gap_rows = [
    {"Router": "Logistic_router",
     "R_train": round(_risk(log_train), 4),
     "R_test":  round(_risk(log_test),  4),
     "Gap":     round(_risk(log_test) - _risk(log_train), 4)},
    {"Router": "LightGBM_router",
     "R_train": round(_risk(gbm_train), 4),
     "R_test":  round(_risk(gbm_test),  4),
     "Gap":     round(_risk(gbm_test) - _risk(gbm_train), 4)},
]
gap_df = pd.DataFrame(gap_rows)
gap_df.to_csv(OUT / "tables" / "router_train_test_gap.csv", index=False)
print(gap_df)
""")


# ---------------------------------------------------------------------------
# 22. Phase B.4 header
# ---------------------------------------------------------------------------
md("""---
## Phase B.4 -- MIRAGE / MedRAG downstream

Two biomedical QA tasks (PubMedQA-labeled, BioASQ-Y/N) crossed with four
retrieval settings (`None (closed-book)`, `BM25`, `BGE-small`,
`BGE-small+BGE-reranker`) and two generators (flan-t5-base; optionally
Llama-3.2-3B-Instruct in 4-bit).

If you do not have access to gated `meta-llama/Llama-3.2-3B-Instruct`,
Llama is silently skipped and only flan-t5-base results are reported.
Run `huggingface-cli login` and accept the licence at
<https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct> to enable Llama.
""")


# ---------------------------------------------------------------------------
# 23. Load PubMedQA + BioASQ-Y/N
# ---------------------------------------------------------------------------
code("""# B.4.a: load PubMedQA-labeled and BioASQ-Y/N.
from datasets import load_dataset

pq = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")
print(f"PubMedQA-labeled: {len(pq)}")
pubmedqa = []
for r in pq:
    pubmedqa.append({
        "qid":               str(r["pubid"]),
        "question":          r["question"],
        "answer":            r["final_decision"].lower().strip(),
        "context_passages":  r["context"]["contexts"],
    })


try:
    ba = load_dataset("kroshan/BioASQ", split="train")
    bioasq_yn = []
    rng_yn = np.random.default_rng(42)
    for r in ba:
        if r.get("type", "").lower() == "yesno" and r.get("answer", "").lower() in ("yes", "no"):
            bioasq_yn.append({
                "qid":      r.get("id", str(len(bioasq_yn))),
                "question": r["body"],
                "answer":   r["answer"].lower(),
            })
    if len(bioasq_yn) > 500:
        idx = rng_yn.choice(len(bioasq_yn), size=500, replace=False)
        bioasq_yn = [bioasq_yn[i] for i in sorted(idx.tolist())]
    print(f"BioASQ-Y/N: {len(bioasq_yn)}")
except Exception as exc:
    print(f"BioASQ-Y/N load failed ({exc}); skipping that task.")
    bioasq_yn = []
""")


# ---------------------------------------------------------------------------
# 24. Build retrieval contexts
# ---------------------------------------------------------------------------
code("""# B.4.b: build top-5 evidence passages per question for each retrieval setting.

def build_pubmedqa_index():
    docs = {}
    for r in pubmedqa:
        for ci, passage in enumerate(r["context_passages"]):
            docs[f"{r['qid']}_p{ci}"] = {"title": "", "text": passage,
                                          "owner_qid": r["qid"]}
    return docs


pubmedqa_corpus = build_pubmedqa_index()
pq_doc_ids   = list(pubmedqa_corpus.keys())
pq_doc_texts = [pubmedqa_corpus[d]["text"] for d in pq_doc_ids]
print(f"PubMedQA pseudo-corpus: {len(pq_doc_ids):,} passages")


def topk_for_questions(rows, retriever_name, k=5):
    queries = {r["qid"]: r["question"] for r in rows}
    if retriever_name == "BM25":
        res = _bm25_run(pq_doc_ids, pq_doc_texts, queries, top_k=k)
    elif retriever_name == "BGE-small":
        res = _dense_run(pq_doc_ids, pq_doc_texts, queries, get_bge(), top_k=k)
    elif retriever_name == "BGE-small+BGE-reranker":
        first = _dense_run(pq_doc_ids, pq_doc_texts, queries, get_bge(), top_k=100)
        cand = {qid: list(first[qid].keys()) for qid in queries}
        rer = cross_encoder_rerank(queries, pubmedqa_corpus, cand, reranker_model,
                                    batch_size=RERANK_BATCH)
        res = {qid: dict(list(rer[qid].items())[:k]) for qid in queries}
    else:
        raise ValueError(retriever_name)
    return {qid: [pubmedqa_corpus[d]["text"] for d in res[qid]] for qid in queries}
""")


# ---------------------------------------------------------------------------
# 25. Generators + evaluation
# ---------------------------------------------------------------------------
code("""# B.4.c: load generators and run flan-t5-base + (optionally) Llama-3.2-3B-Instruct.
import re as _re

def _ensure(pkgs):
    try:
        for p in pkgs:
            importlib.import_module(p.split(">=")[0].split("==")[0])
    except ImportError:
        _pip(pkgs)

_ensure(["bitsandbytes>=0.43.0", "accelerate>=0.32.0", "sentencepiece"])

from transformers import (AutoTokenizer, AutoModelForCausalLM,
                           AutoModelForSeq2SeqLM, BitsAndBytesConfig)


def load_llama():
    name = "meta-llama/Llama-3.2-3B-Instruct"
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
    tok = AutoTokenizer.from_pretrained(name)
    mod = AutoModelForCausalLM.from_pretrained(name, quantization_config=bnb,
                                                device_map="auto")
    return tok, mod


def load_flan():
    name = "google/flan-t5-base"
    tok = AutoTokenizer.from_pretrained(name)
    mod = AutoModelForSeq2SeqLM.from_pretrained(name).to(device)
    return tok, mod


PROMPT_TMPL = (
    "You are a biomedical question-answering assistant.\\n"
    "Answer with one of: yes, no, maybe.\\n"
    "Question: {q}\\n"
    "Evidence:\\n{ctx}\\n"
    "Final answer:"
)


def _format_prompt(q, passages):
    ctx = "\\n".join(f"[{i+1}] {p}" for i, p in enumerate(passages or []))
    if not ctx:
        ctx = "(no retrieved evidence)"
    return PROMPT_TMPL.format(q=q, ctx=ctx)


def _normalize_answer(text):
    t = text.strip().lower()
    for label in ("yes", "no", "maybe"):
        if _re.search(rf"\\b{label}\\b", t):
            return label
    return "unknown"


def _gen_llama(tok, mod, prompt, max_new=8):
    msgs = [{"role": "user", "content": prompt}]
    inp = tok.apply_chat_template(msgs, return_tensors="pt", add_generation_prompt=True).to(mod.device)
    with torch.no_grad():
        out = mod.generate(inp, max_new_tokens=max_new, do_sample=False)
    txt = tok.decode(out[0, inp.shape[-1]:], skip_special_tokens=True)
    return _normalize_answer(txt)


def _gen_flan(tok, mod, prompt, max_new=4):
    inp = tok(prompt, return_tensors="pt", truncation=True, max_length=1024).to(mod.device)
    with torch.no_grad():
        out = mod.generate(**inp, max_new_tokens=max_new, do_sample=False)
    return _normalize_answer(tok.decode(out[0], skip_special_tokens=True))


ROWS, PRED_ROWS = [], []
RETRIEVERS = ["None (closed-book)", "BM25", "BGE-small", "BGE-small+BGE-reranker"]
TASKS = [("PubMedQA-labeled", pubmedqa)]
if bioasq_yn:
    TASKS.append(("BioASQ-Y/N", bioasq_yn))


print("Loading flan-t5-base ...")
ft_tok, ft_mod = load_flan()
print("Loading Llama-3.2-3B-Instruct (4-bit) ...")
try:
    ll_tok, ll_mod = load_llama()
    HAVE_LLAMA = True
except Exception as exc:
    print(f"  Llama-3.2 unavailable ({type(exc).__name__}: {exc}).")
    print("  Falling back to flan-t5-base only. Run `huggingface-cli login` "
          "and accept the licence at "
          "https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct to enable.")
    ll_tok = ll_mod = None
    HAVE_LLAMA = False


for task_name, task_rows in TASKS:
    contexts = {"None (closed-book)": {r["qid"]: [] for r in task_rows}}
    if task_name == "PubMedQA-labeled":
        for retr in ["BM25", "BGE-small", "BGE-small+BGE-reranker"]:
            print(f"  retrieval[{retr}] ...", flush=True)
            contexts[retr] = topk_for_questions(task_rows, retr, k=5)
    else:
        for retr in ["BM25", "BGE-small", "BGE-small+BGE-reranker"]:
            print(f"  retrieval[{retr}] (BioASQ corpus) ...", flush=True)
            queries = {r["qid"]: r["question"] for r in task_rows}
            if retr == "BM25":
                top = _bm25_run(doc_ids_b, doc_texts_b, queries, top_k=5)
            elif retr == "BGE-small":
                top = _dense_run(doc_ids_b, doc_texts_b, queries, get_bge(), top_k=5)
            else:
                first = _dense_run(doc_ids_b, doc_texts_b, queries, get_bge(), top_k=100)
                cand = {qid: list(first[qid].keys()) for qid in queries}
                rer = cross_encoder_rerank(queries, bioasq_corpus, cand, reranker_model,
                                            batch_size=RERANK_BATCH)
                top = {qid: dict(list(rer[qid].items())[:5]) for qid in queries}
            contexts[retr] = {qid: [(bioasq_corpus[d].get("title", "") + " " +
                                     bioasq_corpus[d].get("text", "")).strip()
                                    for d in top[qid]] for qid in queries}

    _gens = [("flan-t5-base", ft_tok, ft_mod, _gen_flan)]
    if HAVE_LLAMA:
        _gens.append(("Llama-3.2-3B-Instruct", ll_tok, ll_mod, _gen_llama))

    for retr in RETRIEVERS:
        for gen_name, tok, mod, gen_fn in _gens:
            correct = total = 0
            for r in tqdm(task_rows, desc=f"{task_name} | {retr} | {gen_name}", leave=False):
                ctx = contexts[retr].get(r["qid"], [])
                ans = gen_fn(tok, mod, _format_prompt(r["question"], ctx))
                gold = r["answer"].lower()
                if task_name == "BioASQ-Y/N" and ans == "maybe":
                    ans = "yes"
                if ans == gold:
                    correct += 1
                total += 1
                PRED_ROWS.append({"task": task_name, "retriever": retr,
                                   "generator": gen_name, "qid": r["qid"],
                                   "gold": gold, "prediction": ans})
            acc = correct / total if total else 0.0
            ROWS.append({"Task": task_name, "Retriever": retr,
                          "Generator": gen_name,
                          "Accuracy": round(acc, 4),
                          "Status": "done"})
            print(f"  {task_name} | {retr} | {gen_name}: acc = {acc:.4f}")


mirage_df = pd.DataFrame(ROWS)
mirage_df.to_csv(OUT / "tables" / "mirage_accuracy.csv", index=False)
print(mirage_df)

pred_df = pd.DataFrame(PRED_ROWS)
pred_df.to_parquet(CACHE_DIR / "mirage_predictions.parquet", index=False)
print(f"Predictions cached -> {CACHE_DIR / 'mirage_predictions.parquet'}")
""")


# ---------------------------------------------------------------------------
# 26. Summary header
# ---------------------------------------------------------------------------
md("""---
## Summary -- final artefacts

The next cell prints every CSV that was written by this notebook so you
can verify the full set before downloading them off Drive.
""")


# ---------------------------------------------------------------------------
# 27. Display all results
# ---------------------------------------------------------------------------
code("""# Final summary: list and print every CSV under OUT/tables/.
import pandas as pd
from pathlib import Path

print(f"All artefacts under: {OUT / 'tables'}\\n")
for p in sorted((OUT / 'tables').glob('*.csv')):
    print("=" * 78)
    print(p.name)
    print("=" * 78)
    try:
        print(pd.read_csv(p).to_string(index=False))
    except Exception as exc:
        print(f"(could not parse: {exc})")
    print()

print("Cache dir:")
for p in sorted((OUT / 'cache').glob('*')):
    print(f"  {p.name}  ({p.stat().st_size:,} bytes)")
""")


# ---------------------------------------------------------------------------
# Emit notebook
# ---------------------------------------------------------------------------
nb = {
    "cells": CELLS,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
        "accelerator": "GPU",
        "colab": {"provenance": []},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUT_PATH = Path(__file__).parent / "run_on_colab.ipynb"
OUT_PATH.write_text(json.dumps(nb, indent=1))
print(f"wrote {OUT_PATH}  ({len(CELLS)} cells)")

"""Phase B: BioASQ retrieval, reranker, per-query router, MIRAGE generation.

Writes:
    multi_dataset_ndcg10_v2.csv   (BioASQ row backfilled here)
    bioasq_paired_bootstrap.csv
    reranker_ndcg10.csv
    router_test.csv, router_feature_importance.csv, router_train_test_gap.csv
    mirage_accuracy.csv

Run:  python scripts/phase_b.py
"""

import json
import math
import pickle
import re
import sys
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rageval.data import load_beir, load_bioasq
from rageval.retrieval import (
    CACHE, FEEDBACK2, RESULTS,
    bm25, dense, splade, medcpt, hybrid, evaluate, prep_corpus, ndcg,
    cross_encoder_rerank, medcpt_ce_rerank, paired_bootstrap,
    free_cuda, device,
    load_bge, load_bge_reranker, load_e5, load_medcpt, load_medcpt_ce,
    load_minilm, load_splade,
)


# ---- B.1  BioASQ-subset retrieval ----

SEED = 42
TARGET = 50_000
SAMPLE = 500


def build_bioasq_subset():
    """Load BioASQ, build qrel-union+distractors, sample 500 queries,
    run all six retrievers, cache for B.2."""
    print("Loading BioASQ ...")
    (corpus, queries, qrels), source = load_bioasq()
    print(f"  source: {source}  full corpus = {len(corpus):,}")

    rng = np.random.default_rng(SEED)
    qrel_union = set()
    for rels in qrels.values():
        qrel_union.update(rels)
    qrel_union &= set(corpus)
    remaining = list(set(corpus) - qrel_union)
    n_dist = max(0, min(TARGET - len(qrel_union), 10_000, len(remaining)))
    distractors = (
        rng.choice(remaining, size=n_dist, replace=False).tolist()
        if n_dist else []
    )
    subset_ids = sorted(qrel_union | set(distractors))
    print(f"  qrel_union={len(qrel_union):,}  distractors={n_dist:,}  "
          f"subset={len(subset_ids):,}")

    out_dir = RESULTS / "feedback2" / "datasets"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "bioasq_subset_doc_ids.json").write_text(json.dumps({
        "seed": SEED, "source": source,
        "n_qrel_union": len(qrel_union), "n_distractors": n_dist,
        "n_total": len(subset_ids), "doc_ids": subset_ids,
    }))

    sub = set(subset_ids)
    sub_corpus = {d: corpus[d] for d in subset_ids}
    sub_qrels = {
        q: {d: r for d, r in rels.items() if d in sub}
        for q, rels in qrels.items()
    }
    sub_qrels = {q: r for q, r in sub_qrels.items() if r}

    rng = np.random.default_rng(SEED)
    eligible = sorted(sub_qrels)
    sampled = sorted(rng.choice(
        eligible, size=min(SAMPLE, len(eligible)), replace=False).tolist())
    sub_queries = {q: queries[q] for q in sampled if q in queries}
    sub_qrels = {q: sub_qrels[q] for q in sub_queries}
    print(f"  sampled {len(sub_queries)} queries")

    doc_ids, doc_texts = prep_corpus(sub_corpus)

    # six-retriever sweep
    results = {}
    print("\n  BM25 ...")
    results["BM25"] = bm25(doc_ids, doc_texts, sub_queries)

    print("  Dense (MiniLM) ...")
    m = load_minilm()
    results["Dense"] = dense(doc_ids, doc_texts, sub_queries, m)
    del m
    free_cuda()

    print("  BGE-small ...")
    m = load_bge()
    results["BGE-small"] = dense(doc_ids, doc_texts, sub_queries, m)
    del m
    free_cuda()

    print("  E5-small ...")
    m = load_e5()
    results["E5-small"] = dense(
        doc_ids, doc_texts, sub_queries, m,
        qpfx="query: ", dpfx="passage: ",
    )
    del m
    free_cuda()

    print("  SPLADE ...")
    s_tok, s_mod = load_splade()
    results["SPLADE"] = splade(doc_ids, doc_texts, sub_queries, s_tok, s_mod)
    del s_tok, s_mod
    free_cuda()

    print("  MedCPT ...")
    q_tok, q_mod, a_tok, a_mod = load_medcpt()
    results["MedCPT"] = medcpt(
        doc_ids, doc_texts, sub_queries, q_tok, q_mod, a_tok, a_mod)
    del q_tok, q_mod, a_tok, a_mod
    free_cuda()

    # cache for B.2
    with open(CACHE / "bioasq_results.pkl", "wb") as fh:
        pickle.dump({
            "results": results, "qrels": sub_qrels,
            "queries": sub_queries, "corpus": sub_corpus,
            "doc_ids": doc_ids, "doc_texts": doc_texts,
            "qrel_union_n": len(qrel_union),
            "n_distractors": n_dist,
        }, fh)

    # Aggregate metrics + backfill the BioASQ row in multi_dataset_ndcg10_v2.csv
    evals = {n: evaluate(r, sub_qrels) for n, r in results.items()}
    md_path = FEEDBACK2 / "multi_dataset_ndcg10_v2.csv"
    md = pd.read_csv(md_path)
    mask = md["Dataset"] == "BioASQ-subset"
    for col, name in [
        ("BM25 nDCG@10", "BM25"), ("Dense nDCG@10", "Dense"),
        ("BGE-small nDCG@10", "BGE-small"),
        ("E5-small nDCG@10", "E5-small"),
        ("SPLADE nDCG@10", "SPLADE"),
        ("MedCPT nDCG@10", "MedCPT"),
    ]:
        md.loc[mask, col] = round(evals[name]["aggregate"]["nDCG@10"], 4)
    md.loc[mask, "# Docs"] = len(doc_ids)
    md.loc[mask, "# Queries"] = len(sub_queries)
    md.loc[mask, "Notes"] = (
        f"qrel-union ({len(qrel_union):,}) + {n_dist:,} distractors, "
        f"seed={SEED}, {len(sub_queries)}-query sample"
    )
    md.to_csv(md_path, index=False)
    print("\n  updated multi_dataset_ndcg10_v2.csv (BioASQ-subset row)")

    # Per-query nDCG@10 parquet (used by the router later)
    qids = sorted(sub_queries)
    pq = {"qid": qids}
    for name, ev in evals.items():
        pq[name] = [ev["per_query"][q]["nDCG@10"] for q in qids]
    pd.DataFrame(pq).to_parquet(CACHE / "bioasq_perquery.parquet", index=False)

    # paired bootstrap MedCPT - BM25
    a = np.array([evals["MedCPT"]["per_query"][q]["nDCG@10"] for q in qids])
    b = np.array([evals["BM25"]["per_query"][q]["nDCG@10"] for q in qids])
    ci = paired_bootstrap(a, b)
    pd.DataFrame([{
        "contrast": "MedCPT - BM25", "dataset": "BioASQ-subset",
        "n_queries": len(qids),
        "mean_diff": round(ci["mean_diff"], 4),
        "ci_lo": round(ci["ci_lo"], 4),
        "ci_hi": round(ci["ci_hi"], 4),
        "p_a_gt_b": round(ci["p_a_gt_b"], 3),
    }]).to_csv(FEEDBACK2 / "bioasq_paired_bootstrap.csv", index=False)
    print(f"  MedCPT-BM25 = {ci['mean_diff']:+.4f} "
          f"[{ci['ci_lo']:+.4f}, {ci['ci_hi']:+.4f}]")


# ---- B.2  Cross-encoder reranking ----

def run_rerankers():
    """BGE cross-encoder over the strongest first stage per dataset, plus
    in-domain MedCPT-CE on the two biomedical sets."""
    reranker = load_bge_reranker()
    free_cuda()
    rows = []
    pq_rows = []  # long-format per-query nDCG@10 for first-stage AND reranked

    def agg_and_pq(res, qrels):
        ev = evaluate(res, qrels)
        return ev["aggregate"]["nDCG@10"], {
            qid: float(m["nDCG@10"]) for qid, m in ev["per_query"].items()
        }

    def record_pq(ds, fs_label, system, k, pq):
        for qid, v in pq.items():
            pq_rows.append({
                "Dataset": ds, "FirstStage": fs_label, "System": system,
                "k": k, "qid": qid, "nDCG@10": round(v, 6),
            })

    def do_rerank(ds, fs_label, fs_res, qrels, corpus, queries, label):
        cand = {q: list(fs_res[q]) for q in queries}
        rer = cross_encoder_rerank(queries, corpus, cand, reranker)
        free_cuda()
        fs_n, fs_pq = agg_and_pq(fs_res, qrels)
        rer_n, rer_pq = agg_and_pq(rer, qrels)
        print(f"  [{ds}] {label}: {fs_n:.4f} -> {rer_n:.4f}  "
              f"(Δ={rer_n - fs_n:+.4f})")
        rows.append({
            "Dataset": ds, "FirstStage": fs_label,
            "FirstStage_nDCG10": round(fs_n, 4),
            "RerankedSystem": label,
            "Reranked_nDCG10": round(rer_n, 4),
            "Delta": round(rer_n - fs_n, 4), "Status": "done",
        })
        record_pq(ds, fs_label, fs_label + " (first-stage)", 100, fs_pq)
        record_pq(ds, fs_label, label, 100, rer_pq)

    # NFCorpus -- two first stages
    print("\n[NFCorpus]")
    corpus, queries, qrels = load_beir("nfcorpus", split="test")
    doc_ids, doc_texts = prep_corpus(corpus)
    bge = load_bge()
    nf_fs_bge = dense(doc_ids, doc_texts, queries, bge)
    del bge
    free_cuda()
    nf_fs_bm25 = bm25(doc_ids, doc_texts, queries)
    do_rerank("NFCorpus", "BGE-small", nf_fs_bge, qrels, corpus, queries,
              "BGE-small+BGE-reranker@100")
    do_rerank("NFCorpus", "BM25", nf_fs_bm25, qrels, corpus, queries,
              "BM25+BGE-reranker@100")
    nf_corpus, nf_qrels, nf_queries = corpus, qrels, queries

    # BioASQ (from B.1 cache)
    print("\n[BioASQ-subset]")
    cache = pickle.load(open(CACHE / "bioasq_results.pkl", "rb"))
    ba_fs_bm25 = cache["results"]["BM25"]
    do_rerank("BioASQ-subset", "BM25", ba_fs_bm25, cache["qrels"],
              cache["corpus"], cache["queries"], "BM25+BGE-reranker@100")

    # TREC-COVID with E5
    print("\n[TREC-COVID]")
    corpus, queries, qrels = load_beir("trec-covid", split="test")
    doc_ids, doc_texts = prep_corpus(corpus)
    e5 = load_e5()
    fs = dense(doc_ids, doc_texts, queries, e5,
               qpfx="query: ", dpfx="passage: ")
    del e5
    free_cuda()
    do_rerank("TREC-COVID", "E5-small", fs, qrels, corpus, queries,
              "E5-small+BGE-reranker@100")

    # SciFact + ArguAna with BGE
    for ds_name, beir_id in [("SciFact", "scifact"), ("ArguAna", "arguana")]:
        print(f"\n[{ds_name}]")
        corpus, queries, qrels = load_beir(beir_id, split="test")
        doc_ids, doc_texts = prep_corpus(corpus)
        bge = load_bge()
        fs = dense(doc_ids, doc_texts, queries, bge)
        del bge
        free_cuda()
        do_rerank(ds_name, "BGE-small", fs, qrels, corpus, queries,
                  "BGE-small+BGE-reranker@100")

    # MedCPT cross-encoder on the two biomedical sets
    print("\n[MedCPT-CE]")
    try:
        ce_tok, ce_mod = load_medcpt_ce()
    except Exception as e:
        print(f"  MedCPT-CE unavailable: {e}")
    else:
        # NFCorpus, BGE first stage
        cand = {q: list(nf_fs_bge[q]) for q in nf_queries}
        rer = medcpt_ce_rerank(nf_queries, nf_corpus, cand, ce_tok, ce_mod)
        fs_n, _ = agg_and_pq(nf_fs_bge, nf_qrels)
        rer_n, rer_pq = agg_and_pq(rer, nf_qrels)
        rows.append({
            "Dataset": "NFCorpus", "FirstStage": "BGE-small",
            "FirstStage_nDCG10": round(fs_n, 4),
            "RerankedSystem": "BGE-small+MedCPT-CE@100",
            "Reranked_nDCG10": round(rer_n, 4),
            "Delta": round(rer_n - fs_n, 4), "Status": "done",
        })
        record_pq("NFCorpus", "BGE-small",
                  "BGE-small+MedCPT-CE@100", 100, rer_pq)
        print(f"  [NFCorpus] BGE-small+MedCPT-CE@100: "
              f"{fs_n:.4f} -> {rer_n:.4f}  (Δ={rer_n - fs_n:+.4f})")

        # BioASQ, BM25 first stage
        cand = {q: list(ba_fs_bm25[q]) for q in cache["queries"]}
        rer = medcpt_ce_rerank(
            cache["queries"], cache["corpus"], cand, ce_tok, ce_mod)
        fs_n, _ = agg_and_pq(ba_fs_bm25, cache["qrels"])
        rer_n, rer_pq = agg_and_pq(rer, cache["qrels"])
        rows.append({
            "Dataset": "BioASQ-subset", "FirstStage": "BM25",
            "FirstStage_nDCG10": round(fs_n, 4),
            "RerankedSystem": "BM25+MedCPT-CE@100",
            "Reranked_nDCG10": round(rer_n, 4),
            "Delta": round(rer_n - fs_n, 4), "Status": "done",
        })
        record_pq("BioASQ-subset", "BM25",
                  "BM25+MedCPT-CE@100", 100, rer_pq)
        print(f"  [BioASQ-subset] BM25+MedCPT-CE@100: "
              f"{fs_n:.4f} -> {rer_n:.4f}  (Δ={rer_n - fs_n:+.4f})")
        del ce_tok, ce_mod
        free_cuda()

    pd.DataFrame(rows).to_csv(FEEDBACK2 / "reranker_ndcg10.csv", index=False)
    pd.DataFrame(pq_rows).to_csv(
        FEEDBACK2 / "reranker_per_query_ndcg10.csv", index=False)
    print(f"\nwrote {FEEDBACK2 / 'reranker_ndcg10.csv'}")
    print(f"wrote {FEEDBACK2 / 'reranker_per_query_ndcg10.csv'} "
          f"({len(pq_rows)} rows)")


# ---- B.2b  Reranker depth ablation + paired bootstrap CIs ----
#
# Reuses the same six (dataset, first-stage) pairs as run_rerankers, but
# also sweeps the candidate-pool depth k in {10, 20, 50, 100} and stores
# per-query nDCG@10 so we can compute paired bootstrap CIs of
# Δ = nDCG@10(rerank) − nDCG@10(first-stage).
#
# The depth sweep tells us whether a negative Δ at k=100 is a task
# mismatch (negative at every depth) or a recall-ceiling artefact
# (negative only at deep k, harmless at small k). The paired CIs tell us
# whether each Δ is significantly different from zero per dataset.

DEPTHS = (10, 20, 50, 100)


def _per_q_ndcg10(res, qrels):
    return {
        qid: ndcg(sorted(r, key=r.get, reverse=True), qrels.get(qid, {}), 10)
        for qid, r in res.items()
    }


def _top_k_candidates(fs_res, k):
    out = {}
    for qid, scores in fs_res.items():
        ordered = sorted(scores, key=scores.get, reverse=True)[:k]
        out[qid] = ordered
    return out


def rerank_depth_and_ci(depths=DEPTHS, B=10_000):
    """Depth sweep + paired bootstrap CIs for the BGE-reranker rows.

    Produces:
        reranker_depth_sweep.csv     wide format -- one row per
                                     (dataset, first-stage, k) with
                                     aggregate nDCG@10 and Delta.
        reranker_paired_bootstrap.csv  one row per (dataset, first-stage, k)
                                     with paired-bootstrap mean diff,
                                     95% CI and P(rerank > first-stage).
    """
    reranker = load_bge_reranker()
    free_cuda()

    sweep_rows = []
    ci_rows = []
    pq_rows = []  # long-format per-query nDCG@10 for every (ds, fs, k)

    def evaluate_combo(ds, fs_label, fs_res, qrels, corpus, queries, reranker_name):
        fs_pq = _per_q_ndcg10(fs_res, qrels)
        fs_agg = float(np.mean(list(fs_pq.values())))
        for k in depths:
            cand = _top_k_candidates(fs_res, k)
            rer = cross_encoder_rerank(queries, corpus, cand, reranker)
            free_cuda()
            rer_pq = _per_q_ndcg10(rer, qrels)
            rer_agg = float(np.mean(list(rer_pq.values())))
            qids = sorted(rer_pq)
            a = np.array([rer_pq[q] for q in qids])
            b = np.array([fs_pq.get(q, 0.0) for q in qids])
            ci = paired_bootstrap(a, b, B=B)
            sweep_rows.append({
                "Dataset": ds, "FirstStage": fs_label,
                "Reranker": reranker_name, "k": k,
                "FirstStage_nDCG10": round(fs_agg, 4),
                "Reranked_nDCG10": round(rer_agg, 4),
                "Delta": round(rer_agg - fs_agg, 4),
                "n_queries": len(qids),
            })
            ci_rows.append({
                "Dataset": ds, "FirstStage": fs_label,
                "Reranker": reranker_name, "k": k,
                "n_queries": len(qids), "B": B,
                "mean_diff": round(ci["mean_diff"], 4),
                "ci_lo": round(ci["ci_lo"], 4),
                "ci_hi": round(ci["ci_hi"], 4),
                "p_a_gt_b": round(ci["p_a_gt_b"], 3),
            })
            for qid in qids:
                pq_rows.append({
                    "Dataset": ds, "FirstStage": fs_label,
                    "Reranker": reranker_name, "k": k, "qid": qid,
                    "first_stage_nDCG@10": round(float(fs_pq.get(qid, 0.0)), 6),
                    "reranked_nDCG@10": round(float(rer_pq[qid]), 6),
                    "delta": round(float(rer_pq[qid] - fs_pq.get(qid, 0.0)), 6),
                })
            print(f"  [{ds}/{fs_label}] k={k:3d}  Δ={rer_agg - fs_agg:+.4f}  "
                  f"CI=[{ci['ci_lo']:+.4f}, {ci['ci_hi']:+.4f}]  "
                  f"P(rerank>FS)={ci['p_a_gt_b']:.2f}")

    # NFCorpus -- BGE-small and BM25 first stages
    print("\n[NFCorpus]")
    corpus, queries, qrels = load_beir("nfcorpus", split="test")
    doc_ids, doc_texts = prep_corpus(corpus)
    bge = load_bge()
    nf_fs_bge = dense(doc_ids, doc_texts, queries, bge)
    del bge
    free_cuda()
    nf_fs_bm25 = bm25(doc_ids, doc_texts, queries)
    evaluate_combo("NFCorpus", "BGE-small", nf_fs_bge, qrels, corpus, queries,
                   "BGE-reranker-base")
    evaluate_combo("NFCorpus", "BM25", nf_fs_bm25, qrels, corpus, queries,
                   "BGE-reranker-base")

    # BioASQ (from B.1 cache)
    print("\n[BioASQ-subset]")
    cache = pickle.load(open(CACHE / "bioasq_results.pkl", "rb"))
    evaluate_combo(
        "BioASQ-subset", "BM25", cache["results"]["BM25"], cache["qrels"],
        cache["corpus"], cache["queries"], "BGE-reranker-base",
    )

    # TREC-COVID with E5
    print("\n[TREC-COVID]")
    corpus, queries, qrels = load_beir("trec-covid", split="test")
    doc_ids, doc_texts = prep_corpus(corpus)
    e5 = load_e5()
    fs = dense(doc_ids, doc_texts, queries, e5,
               qpfx="query: ", dpfx="passage: ")
    del e5
    free_cuda()
    evaluate_combo("TREC-COVID", "E5-small", fs, qrels, corpus, queries,
                   "BGE-reranker-base")

    # SciFact + ArguAna with BGE
    for ds_name, beir_id in [("SciFact", "scifact"), ("ArguAna", "arguana")]:
        print(f"\n[{ds_name}]")
        corpus, queries, qrels = load_beir(beir_id, split="test")
        doc_ids, doc_texts = prep_corpus(corpus)
        bge = load_bge()
        fs = dense(doc_ids, doc_texts, queries, bge)
        del bge
        free_cuda()
        evaluate_combo(ds_name, "BGE-small", fs, qrels, corpus, queries,
                       "BGE-reranker-base")

    del reranker
    free_cuda()

    pd.DataFrame(sweep_rows).to_csv(
        FEEDBACK2 / "reranker_depth_sweep.csv", index=False)
    pd.DataFrame(ci_rows).to_csv(
        FEEDBACK2 / "reranker_paired_bootstrap.csv", index=False)
    pd.DataFrame(pq_rows).to_csv(
        FEEDBACK2 / "reranker_depth_per_query_ndcg10.csv", index=False)
    print(f"\nwrote {FEEDBACK2 / 'reranker_depth_sweep.csv'}")
    print(f"wrote {FEEDBACK2 / 'reranker_paired_bootstrap.csv'}")
    print(f"wrote {FEEDBACK2 / 'reranker_depth_per_query_ndcg10.csv'} "
          f"({len(pq_rows)} rows)")


# ---- B.3  Per-query router on NFCorpus ----

WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9\-]+")
QUESTION_RE = re.compile(
    r"^\s*(what|who|how|why|when|where|is|are|does|do|can|should)\b", re.I)
MED_AFFIX = (
    "mab", "tinib", "vir", "olol", "azepam", "azole", "statin",
    "cycline", "itis", "osis", "emia", "oma", "pathy",
    "cardio", "neuro", "hepato", "renal", "gastro", "pulmonary",
)
ALPHA = 0.40
STRATS = ["BM25", "BGE-small", "Hybrid", "BGE+BGE-reranker"]
FEATURES = ["gap", "len", "tech", "is_question", "med_share", "bm25_top1"]


def _toks(s):
    return [t.lower() for t in WORD_RE.findall(s)]


def _med_share(toks):
    if not toks:
        return 0.0
    return sum(
        any(t.endswith(x) or x in t for x in MED_AFFIX) for t in toks
    ) / len(toks)


def _vocab_gap(qtxt, rels, doc_text_by_id):
    qset = set(_toks(qtxt))
    if not qset:
        return 0.0
    best = 0.0
    for d in rels:
        if d in doc_text_by_id:
            ov = len(qset & set(_toks(doc_text_by_id[d]))) / len(qset)
            best = max(best, ov)
    return 1.0 - best


def _per_q_ndcg(res, qrels):
    return {
        qid: ndcg(sorted(r, key=r.get, reverse=True), qrels.get(qid, {}), 10)
        for qid, r in res.items()
    }


def _bm25_top1(queries, doc_ids, doc_texts):
    """Cheap easy-query signal: top-1 BM25 score for each query."""
    import bm25s
    corpus_tok = bm25s.tokenize(doc_texts, stopwords="en", show_progress=False)
    index = bm25s.BM25(k1=1.5, b=0.75)
    index.index(corpus_tok)
    out = {}
    for qid, q in queries.items():
        q_tok = bm25s.tokenize([q], stopwords="en", show_progress=False)
        try:
            _, sc = index.retrieve(q_tok, k=1, n_threads=1, show_progress=False)
        except TypeError:
            _, sc = index.retrieve(q_tok, k=1)
        out[qid] = float(sc[0][0]) if len(sc) and len(sc[0]) else 0.0
    return out


def run_router():
    splits = {}
    corpus = None
    for s in ("train", "dev", "test"):
        c, q, r = load_beir("nfcorpus", split=s)
        if corpus is None:
            corpus = c
        splits[s] = {"queries": q, "qrels": r}
    doc_ids, doc_texts = prep_corpus(corpus)
    doc_text_by_id = dict(zip(doc_ids, doc_texts))

    # technicality (mean IDF)
    df = Counter()
    for t in doc_texts:
        df.update(set(_toks(t)))
    n = len(doc_ids)
    idf = {tok: math.log((n + 1) / (df[tok] + 1)) + 1.0 for tok in df}

    # BM25 top-1 score per query, all splits
    bm25_top1 = {}
    for s in ("train", "dev", "test"):
        bm25_top1.update(_bm25_top1(splits[s]["queries"], doc_ids, doc_texts))

    # per-query features
    feat_rows = []
    for s in ("train", "dev", "test"):
        for qid, qtxt in splits[s]["queries"].items():
            toks = _toks(qtxt)
            rels = splits[s]["qrels"].get(qid, {})
            feat_rows.append({
                "split": s, "qid": qid,
                "gap": _vocab_gap(qtxt, rels, doc_text_by_id),
                "len": len(toks),
                "tech": float(np.mean([idf.get(t, 1.0) for t in toks])) if toks else 0.0,
                "is_question": int(bool(QUESTION_RE.match(qtxt))),
                "med_share": _med_share(toks),
                "bm25_top1": bm25_top1.get(qid, 0.0),
            })
    feats = pd.DataFrame(feat_rows)

    # per-strategy nDCG@10 across all splits
    bge = load_bge()
    reranker = load_bge_reranker()
    free_cuda()

    rows = []
    for split_name, payload in splits.items():
        bm25_res = bm25(doc_ids, doc_texts, payload["queries"])
        bge_res = dense(doc_ids, doc_texts, payload["queries"], bge)
        hyb_res = hybrid(bm25_res, bge_res, payload["queries"], alpha=ALPHA)
        cand = {q: list(bge_res[q]) for q in payload["queries"]}
        rer_res = cross_encoder_rerank(payload["queries"], corpus, cand, reranker)
        free_cuda()
        bm25_q = _per_q_ndcg(bm25_res, payload["qrels"])
        bge_q = _per_q_ndcg(bge_res, payload["qrels"])
        hyb_q = _per_q_ndcg(hyb_res, payload["qrels"])
        rer_q = _per_q_ndcg(rer_res, payload["qrels"])
        for qid in payload["queries"]:
            rows.append({
                "split": split_name, "qid": qid,
                "BM25": bm25_q.get(qid, 0.0),
                "BGE-small": bge_q.get(qid, 0.0),
                "Hybrid": hyb_q.get(qid, 0.0),
                "BGE+BGE-reranker": rer_q.get(qid, 0.0),
            })
    perq = pd.DataFrame(rows)

    del bge, reranker
    free_cuda()

    # Train logistic + LightGBM routers
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    import lightgbm as lgb

    df_all = feats.merge(perq, on=["split", "qid"], how="inner")
    df_all["label"] = df_all[STRATS].values.argmax(axis=1)

    train_mask = df_all["split"].isin(["train", "dev"])
    test_mask = df_all["split"] == "test"
    X_tr = df_all.loc[train_mask, FEATURES].values
    y_tr = df_all.loc[train_mask, "label"].values
    X_te = df_all.loc[test_mask, FEATURES].values

    log_pipe = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000),
    ).fit(X_tr, y_tr)
    gbm = lgb.LGBMClassifier(
        num_leaves=31, learning_rate=0.05, n_estimators=300,
        objective="multiclass", num_class=4, random_state=42, verbose=-1,
    ).fit(X_tr, y_tr)

    def route(model, X, df_split):
        preds = model.predict(X)
        chosen = df_split[STRATS].values[np.arange(len(df_split)), preds]
        return float(np.mean(chosen))

    test_df = df_all.loc[test_mask].reset_index(drop=True)
    train_df = df_all.loc[train_mask].reset_index(drop=True)
    log_te = route(log_pipe, X_te, test_df)
    gbm_te = route(gbm, X_te, test_df)
    log_tr = route(log_pipe, X_tr, train_df)
    gbm_tr = route(gbm, X_tr, train_df)
    oracle = float(np.mean(test_df[STRATS].values.max(axis=1)))

    pd.DataFrame([
        {"Method": "Always_BM25",
         "nDCG@10": round(float(test_df["BM25"].mean()), 4), "Status": "baseline"},
        {"Method": "Always_BGE_small",
         "nDCG@10": round(float(test_df["BGE-small"].mean()), 4), "Status": "baseline"},
        {"Method": "Static_Hybrid_alpha_0.40",
         "nDCG@10": round(float(test_df["Hybrid"].mean()), 4), "Status": "baseline"},
        {"Method": "Always_BGE_small+BGE_reranker",
         "nDCG@10": round(float(test_df["BGE+BGE-reranker"].mean()), 4),
         "Status": "baseline"},
        {"Method": "Logistic_router",
         "nDCG@10": round(log_te, 4), "Status": "learned"},
        {"Method": "LightGBM_router",
         "nDCG@10": round(gbm_te, 4), "Status": "learned"},
        {"Method": "Oracle_router (upper bound)",
         "nDCG@10": round(oracle, 4), "Status": "oracle"},
    ]).to_csv(FEEDBACK2 / "router_test.csv", index=False)
    print(f"  router: LightGBM={gbm_te:.4f}  Oracle={oracle:.4f}")

    pd.DataFrame({
        "feature": FEATURES,
        "gain": gbm.booster_.feature_importance(importance_type="gain"),
    }).sort_values("gain", ascending=False).to_csv(
        FEEDBACK2 / "router_feature_importance.csv", index=False)

    pd.DataFrame([
        {"Router": "Logistic_router",
         "R_train": round(1 - log_tr, 4),
         "R_test": round(1 - log_te, 4),
         "Gap": round((1 - log_te) - (1 - log_tr), 4)},
        {"Router": "LightGBM_router",
         "R_train": round(1 - gbm_tr, 4),
         "R_test": round(1 - gbm_te, 4),
         "Gap": round((1 - gbm_te) - (1 - gbm_tr), 4)},
    ]).to_csv(FEEDBACK2 / "router_train_test_gap.csv", index=False)

    # Persist per-query: features, per-strategy nDCG@10, oracle label,
    # and router predictions on every split. This is what makes the
    # bootstrap/train-test gap re-computable from disk without re-running
    # the encoders.
    log_preds = log_pipe.predict(df_all[FEATURES].values)
    gbm_preds = gbm.predict(df_all[FEATURES].values)
    perq_out = df_all.copy()
    perq_out["logistic_pred"] = log_preds
    perq_out["lightgbm_pred"] = gbm_preds
    perq_out["logistic_score"] = perq_out[STRATS].values[
        np.arange(len(perq_out)), log_preds]
    perq_out["lightgbm_score"] = perq_out[STRATS].values[
        np.arange(len(perq_out)), gbm_preds]
    perq_out["oracle_score"] = perq_out[STRATS].values.max(axis=1)
    perq_out.to_csv(
        FEEDBACK2 / "router_per_query.csv", index=False)
    print("  wrote router_test, router_feature_importance, "
          "router_train_test_gap, router_per_query")
    print(f"  router_per_query: {len(perq_out)} rows "
          f"({list(perq_out.columns)})")


# ---- B.4  Downstream PubMedQA with flan-t5-base (Llama optional) ----

PROMPT = (
    "You are a biomedical question-answering assistant.\n"
    "Answer with one of: yes, no, maybe.\n"
    "Question: {q}\n"
    "Evidence:\n{ctx}\n"
    "Final answer:"
)


def _normalise(text):
    t = text.strip().lower()
    for label in ("yes", "no", "maybe"):
        if re.search(rf"\b{label}\b", t):
            return label
    return "unknown"


def _format_prompt(q, passages):
    ctx = "\n".join(f"[{i + 1}] {p}" for i, p in enumerate(passages or []))
    return PROMPT.format(q=q, ctx=ctx or "(no retrieved evidence)")


def run_mirage():
    """PubMedQA-labeled with flan-t5-base. Llama is gated; if the user is
    not logged in we silently fall back to flan-t5-base only."""
    import torch
    from datasets import load_dataset

    print("Loading PubMedQA-labeled ...")
    ds = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")

    rows, docs = [], {}
    for r in ds:
        rows.append({
            "qid": str(r["pubid"]),
            "question": r["question"],
            "answer": r["final_decision"].lower().strip(),
            "context": r["context"]["contexts"],
        })
        for ci, p in enumerate(r["context"]["contexts"]):
            docs[f"{r['pubid']}_p{ci}"] = {"title": "", "text": p}

    print(f"  questions = {len(rows)}  passages = {len(docs)}")

    queries = {r["qid"]: r["question"] for r in rows}
    doc_ids = list(docs)
    doc_texts = [docs[d]["text"] for d in doc_ids]

    print("\n  retrieval ...")
    top_bm = bm25(doc_ids, doc_texts, queries, top_k=5)
    bge = load_bge()
    top_bge = dense(doc_ids, doc_texts, queries, bge, top_k=5)
    reranker = load_bge_reranker()
    free_cuda()
    first = dense(doc_ids, doc_texts, queries, bge, top_k=100)
    cand = {q: list(first[q]) for q in queries}
    rer = cross_encoder_rerank(queries, docs, cand, reranker)
    top_rerank = {q: dict(list(rer[q].items())[:5]) for q in queries}
    del bge, reranker
    free_cuda()

    contexts = {
        "None (closed-book)":     {r["qid"]: [] for r in rows},
        "BM25":                   {q: [docs[d]["text"] for d in top_bm[q]] for q in queries},
        "BGE-small":              {q: [docs[d]["text"] for d in top_bge[q]] for q in queries},
        "BGE-small+BGE-reranker": {q: [docs[d]["text"] for d in top_rerank[q]] for q in queries},
    }

    from transformers import (
        AutoTokenizer, AutoModelForSeq2SeqLM,
        AutoModelForCausalLM, BitsAndBytesConfig,
    )

    print("\n  loading flan-t5-base ...")
    dev = device()
    dt = torch.float16 if dev == "cuda" else torch.float32
    ft_tok = AutoTokenizer.from_pretrained("google/flan-t5-base")
    ft_mod = AutoModelForSeq2SeqLM.from_pretrained(
        "google/flan-t5-base", torch_dtype=dt).to(dev)

    def gen_flan(prompt, max_new=4):
        inp = ft_tok(
            prompt, return_tensors="pt", truncation=True, max_length=1024,
        ).to(ft_mod.device)
        with torch.no_grad():
            out = ft_mod.generate(**inp, max_new_tokens=max_new, do_sample=False)
        return _normalise(ft_tok.decode(out[0], skip_special_tokens=True))

    print("  loading Llama-3.2-3B-Instruct (4-bit) ...")
    have_llama = False
    ll_tok = ll_mod = None
    try:
        bnb = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
        ll_tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
        ll_mod = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Llama-3.2-3B-Instruct",
            quantization_config=bnb, device_map="auto")
        have_llama = True
    except Exception as e:
        warnings.warn(f"Llama unavailable ({type(e).__name__}); flan-t5 only.")

    def gen_llama(prompt, max_new=8):
        msgs = [{"role": "user", "content": prompt}]
        inp = ll_tok.apply_chat_template(
            msgs, return_tensors="pt", add_generation_prompt=True,
        ).to(ll_mod.device)
        with torch.no_grad():
            out = ll_mod.generate(inp, max_new_tokens=max_new, do_sample=False)
        return _normalise(ll_tok.decode(
            out[0, inp.shape[-1]:], skip_special_tokens=True))

    scored = []
    per_q = []  # one row per (retriever, generator, question)
    gens = [("flan-t5-base", gen_flan)]
    if have_llama:
        gens.append(("Llama-3.2-3B-Instruct", gen_llama))

    for retr in contexts:
        for gen_name, gen_fn in gens:
            correct = 0
            for r in tqdm(rows, desc=f"{retr} | {gen_name}", leave=False):
                ctx = contexts[retr].get(r["qid"], [])
                pred = gen_fn(_format_prompt(r["question"], ctx))
                ok = int(pred == r["answer"])
                correct += ok
                per_q.append({
                    "Task": "PubMedQA-labeled",
                    "Retriever": retr, "Generator": gen_name,
                    "qid": r["qid"], "predicted": pred,
                    "gold": r["answer"], "correct": ok,
                })
            acc = correct / len(rows)
            scored.append({
                "Task": "PubMedQA-labeled", "Retriever": retr,
                "Generator": gen_name, "Accuracy": round(acc, 3),
                "Status": "done",
            })
            print(f"  {retr:<25} | {gen_name:<22} acc = {acc:.3f}")
            free_cuda()

    pd.DataFrame(scored).to_csv(FEEDBACK2 / "mirage_accuracy.csv", index=False)
    pd.DataFrame(per_q).to_csv(
        FEEDBACK2 / "mirage_per_question.csv", index=False)
    print(f"\nwrote {FEEDBACK2 / 'mirage_accuracy.csv'}")
    print(f"wrote {FEEDBACK2 / 'mirage_per_question.csv'} ({len(per_q)} rows)")


if __name__ == "__main__":
    build_bioasq_subset()
    run_rerankers()
    rerank_depth_and_ci()
    run_router()
    run_mirage()

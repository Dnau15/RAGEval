# Reproducibility guide

How to regenerate every CSV that `main.tex` references, starting from
a clean checkout of the repo.

## Environment

The full pipeline was developed on Google Colab with a Tesla T4 GPU.
A CPU-only machine works for `analysis.py`, but the encoder sweep
(`phase_a.py`, `phase_b.py`) needs at least 12 GB of VRAM.  Python 3.10+.

```bash
pip install -r requirements.txt
```

The Llama generator in `phase_b.py` requires a HuggingFace login
because `meta-llama/Llama-3.2-3B-Instruct` is gated:

```bash
huggingface-cli login
```

If the login is missing, `phase_b.run_mirage` silently falls back to
flan-t5-base only and writes the four PubMedQA rows that succeed.

## Run

Order matters: `analysis.py` writes `hybrid_test_comparison.csv`
which `phase_a.write_n_required` reads, and `phase_b.run_rerankers`
reads the BioASQ pickle that `phase_b.build_bioasq_subset` writes.

```bash
# Full pipeline, in dependency order
python scripts/run_all.py

# Or run each stage on its own
python scripts/analysis.py    # paired boot, regression, subsampling
python scripts/phase_a.py     # first-stage + multi-dataset + n_required + efficiency
python scripts/phase_b.py     # BioASQ + reranker + router + MIRAGE
```

A full T4 run takes 2-3 hours.

## What each script writes

| Script | CSVs | Tables in report |
|---|---|---|
| `analysis.py`  | `split_metrics_baselines.csv`, `split_metrics_with_hybrid.csv`, `hybrid_alpha_sweep_dev.csv`, `hybrid_test_comparison.csv`, `query_subset_resampling.csv`, `query_subset_resampling_summary.csv`, `wilcoxon_dense_bm25.csv`, `vocabulary_gap_features.csv`, `vocabulary_gap_correlations.csv`, `vocabulary_gap_stratification_ndcg10.csv`, `technicality_stratification_ndcg10.csv`, `delta_regression_coefficients.csv` | Tab 6, 7, 12, 13 (first-stage cols), 14, 15 |
| `phase_a.py`   | `nfcorpus_canonical.csv`, `nfcorpus_full_metrics.csv`, `multi_dataset_ndcg10_v2.csv`, `n_required.csv`, `efficiency.csv` | Tab 3, 4, 10, 11 |
| `phase_b.py`   | `multi_dataset_ndcg10_v2.csv` (BioASQ row backfill), `bioasq_paired_bootstrap.csv`, `reranker_ndcg10.csv`, `router_test.csv`, `router_feature_importance.csv`, `router_train_test_gap.csv`, `mirage_accuracy.csv` | Tab 4 BioASQ row, Tab 5, 8, 9, 13 (router cols), 16 |

## Layout

```
RAGEval/
├── REPRODUCE.md
├── main.tex
├── requirements.txt
├── src/rageval/
│   ├── __init__.py
│   ├── retrieval.py        # metrics, retrieval, model loaders
│   └── data.py             # BEIR + BioASQ loaders
├── scripts/
│   ├── analysis.py         # paired bootstrap, Δ regression, stratifications
│   ├── phase_a.py          # first-stage + multi-dataset + n_required + efficiency
│   ├── phase_b.py          # BioASQ + reranker + router + MIRAGE
│   └── run_all.py          # orchestrator
└── notebooks/results/
    ├── feedback2/tables/   # Phase A canonical + Phase B CSVs
    ├── feedback2/cache/    # pickles for cross-script reuse
    ├── feedback2/datasets/ # BioASQ subset manifest
    └── next_stage/tables/  # paired bootstrap, regression, stratifications
```

## Caveats baked into the data

**BioASQ subset is degenerate on most public mirrors.**  The
`rag-mini-bioasq` fallback that `phase_b.build_bioasq_subset` ends
up using has a corpus pool exhausted by the qrel union, so the
sampled subset has zero distractors.  This inflates BM25 by roughly
+0.30 nDCG@10 over the published full-corpus baseline.  The caveat
is documented in `main.tex` Section 4.2.  To get a non-degenerate
run, register at <http://bioasq.org/>, download the BEIR-format
release, and unpack it under `notebooks/datasets/bioasq/` so it
contains `corpus.jsonl`, `queries.jsonl` and `qrels/test.tsv`.  The
loader will then prefer the local directory.

**Calibration drift between Phase A and Phase B first-stage
NFCorpus.**  BM25 NFCorpus = 0.2940 in `nfcorpus_canonical.csv` but
0.3076 inside `reranker_ndcg10.csv` and `router_test.csv`.  Caused
by a tokenizer change in `bm25s` between the two pipelines.  The
Delta and routing columns are calibration-invariant, so the
methodological findings are robust.  Documented in the Section 4.3
calibration paragraph.

**Llama-3.2-3B-Instruct is gated.**  Without a HuggingFace login,
`mirage_accuracy.csv` reports only the four PubMedQA + flan-t5
rows.  The MIRAGE table in the report acknowledges this.

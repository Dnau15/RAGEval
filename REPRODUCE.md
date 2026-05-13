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

## Rebuilding `main.tex` tables from CSVs

`main.tex` does not contain hand-typed numbers. Each data table is
`\input{tables/<name>.tex}` and the `.tex` fragments are generated from
the CSVs in `tables/` by `scripts/build_tables.py`.

```bash
# Re-render .tex fragments from the committed tables/*.csv snapshot
python scripts/build_tables.py

# Or pull fresh CSVs from notebooks/results/ first (after a pipeline rerun),
# overwrite the snapshot, and re-render
python scripts/build_tables.py --refresh
```

The committed `tables/` directory is the source of truth for the
report. The live pipeline output in `notebooks/results/` is gitignored
because it is reproducible; `--refresh` is the bridge between the two.

`tables/datasets_manifest.csv` lists corpus size, query count and the
loader source per dataset; `MODELS.md` pins the HuggingFace
checkpoints used for every dense and cross-encoder model.

## What each script writes

| Script | Aggregate CSVs | Tables in report |
|---|---|---|
| `analysis.py`  | `split_metrics_baselines.csv`, `split_metrics_with_hybrid.csv`, `hybrid_alpha_sweep_dev.csv`, `hybrid_test_comparison.csv`, `query_subset_resampling.csv`, `query_subset_resampling_summary.csv`, `wilcoxon_dense_bm25.csv`, `vocabulary_gap_features.csv`, `vocabulary_gap_correlations.csv`, `vocabulary_gap_stratification_ndcg10.csv`, `technicality_stratification_ndcg10.csv`, `delta_regression_coefficients.csv` | Tab 6, 7, 12, 13 (first-stage cols), 14, 15 |
| `phase_a.py`   | `nfcorpus_canonical.csv`, `nfcorpus_full_metrics.csv`, `multi_dataset_ndcg10_v2.csv`, `n_required.csv`, `efficiency.csv` | Tab 3, 4, 10, 11 |
| `phase_b.py`   | `multi_dataset_ndcg10_v2.csv` (BioASQ row backfill), `bioasq_paired_bootstrap.csv`, `reranker_ndcg10.csv`, `reranker_depth_sweep.csv`, `reranker_paired_bootstrap.csv`, `router_test.csv`, `router_feature_importance.csv`, `router_train_test_gap.csv`, `mirage_accuracy.csv` | Tab 4 BioASQ row, Tab 5, 5a/5b (depth + CI), 8, 9, 13 (router cols), 16 |

### Per-query outputs

Per-query nDCG@10 (and per-question correctness for the MIRAGE
section) is persisted as long-format CSVs next to the aggregates.
These files are what makes the paired bootstrap, the train--test gap
and the depth ablation re-computable from disk without rerunning the
encoders.

| File | Written by | Schema |
|---|---|---|
| `nfcorpus_per_query_ndcg10.csv` | `analysis.split_metrics_and_alpha` | `split, method, qid, nDCG@10` -- BM25 / Dense / Hybrid on the train, dev and test splits of NFCorpus. |
| `nfcorpus_union_per_query_ndcg10.csv` | `analysis.boot_and_subsampling` | `split, method, qid, nDCG@10` -- BM25 / Dense / Hybrid on the 3,237-query train+dev+test union (used by the query-subset bootstrap). |
| `vocabulary_gap_features.csv` | `analysis.regression_and_stratifications` | `qid, gap, len, tech, BM25, Dense, Hybrid, delta` -- per-query features and per-query nDCG@10 for BM25 / Dense / Hybrid on the NFCorpus test split. (Previously this CSV was written before the nDCG columns were joined; it now contains them.) |
| `multi_dataset_per_query_ndcg10.csv` | `phase_a.first_stage_and_multi_dataset` | `dataset, method, qid, nDCG@10` -- every first-stage row in Table 4 expanded to per-query level. |
| `bioasq_perquery.parquet` | `phase_b.build_bioasq_subset` | `qid, BM25, Dense, BGE-small, E5-small, SPLADE, MedCPT` -- BioASQ-subset per-query nDCG@10 (used by the router). |
| `reranker_per_query_ndcg10.csv` | `phase_b.run_rerankers` | `Dataset, FirstStage, System, k, qid, nDCG@10` -- first-stage AND reranked per-query for every row in Table 5 (at $k=100$). |
| `reranker_depth_per_query_ndcg10.csv` | `phase_b.rerank_depth_and_ci` | `Dataset, FirstStage, Reranker, k, qid, first_stage_nDCG@10, reranked_nDCG@10, delta` -- per-query rows for the depth ablation at $k \in \{10, 20, 50, 100\}$. |
| `router_per_query.csv` | `phase_b.run_router` | One row per (split, qid): six features, four strategy nDCG@10 columns, oracle label, oracle nDCG@10, logistic / LightGBM predictions and the nDCG@10 each router actually obtains. |
| `mirage_per_question.csv` | `phase_b.run_mirage` | `Task, Retriever, Generator, qid, predicted, gold, correct` -- one row per (retriever, generator, question). |

Aggregate metrics, bootstrap CIs, dataset manifests
(`tables/datasets_manifest.csv`) and model versions (`MODELS.md`) are
saved alongside.

## Layout

```
RAGEval/
├── REPRODUCE.md
├── MODELS.md               # pinned HF checkpoints for every model
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
│   ├── build_tables.py     # CSV → .tex fragments, used by main.tex
│   └── run_all.py          # orchestrator
├── tables/                 # committed: CSV snapshot + .tex fragments
│   ├── *.csv               # one per main.tex table
│   ├── *.tex               # generated by build_tables.py
│   └── datasets_manifest.csv
└── notebooks/results/      # gitignored: live pipeline output
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
+0.30 nDCG@10 over the published full-corpus baseline.  The
BioASQ-subset row in `multi_dataset_ndcg10_v2.csv` and the BioASQ
rows in `reranker_ndcg10.csv` are kept as a stress case and are
**not** used as biomedical evidence in the report (see `main.tex`
Section 4.2 and the dagger in Table 4).  To get a non-degenerate run,
register at <http://bioasq.org/>, download the BEIR-format release,
and unpack it under `notebooks/datasets/bioasq/` so it contains
`corpus.jsonl`, `queries.jsonl` and `qrels/test.tsv`.  The loader
will then prefer the local directory.

**PubMedQA-labeled candidate pool is tiny.**  PubMedQA-labeled ships
ten candidate passages per question, so all three retrievers in
`mirage_accuracy.csv` saturate at ~0.56.  The retrieval comparison
is not interpretable; only the closed-book vs.\ retrieval contrast
is reported as evidence in `main.tex` Section 5.

**Router uses one retrospective feature.**  The vocabulary gap
feature in `phase_b.run_router` is computed over the qrels and is
not available at deployment time.  The router result in
`router_test.csv` is therefore a near-oracle probe, not a deployment
number; the negative finding (learned router does not beat Static
Hybrid) still holds.  See `main.tex` Section 4.6 for the disclaimer.

**Reranker depth + paired CIs need a Colab rerun.**
`phase_b.rerank_depth_and_ci` sweeps the candidate depth
$k \in \{10, 20, 50, 100\}$ for every (dataset, first-stage) pair and
writes paired bootstrap CIs at $k=100$.  The committed
`tables/reranker_depth_sweep.csv` and
`tables/reranker_paired_bootstrap.csv` carry only the $k=100$ point
estimates that we already had from `reranker_ndcg10.csv`; the
small-$k$ rows and the CI columns are filled by the function on a
Colab T4.  After the rerun, refresh and rebuild the report:

```bash
python scripts/build_tables.py --refresh
pdflatex main.tex
```

`main.tex` Section 4.4 explains the depth-ablation diagnostic for
the ArguAna result: the task-mismatch claim is justified only if
$\Delta$ stays negative across all $k$.

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

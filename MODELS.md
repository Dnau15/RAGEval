# Model checkpoints

Every model used in this report. HuggingFace IDs are pinned; if the
upstream weights move, results may shift, so the revision SHA in
`huggingface_hub` should be recorded for a strict rerun.

## First-stage retrievers

| Role | HuggingFace ID | Params | Loader in `rageval.retrieval` |
|---|---|---|---|
| BM25 | n/a (`bm25s` library) | n/a | `bm25` (`k1=1.5, b=0.75`) |
| Dense (MiniLM) | `sentence-transformers/all-MiniLM-L6-v2` | 22M | `load_minilm` |
| BGE-small | `BAAI/bge-small-en-v1.5` | 33M | `load_bge` |
| E5-small | `intfloat/e5-small-v2` | 33M | `load_e5` |
| SPLADE | `naver/splade-cocondenser-ensembledistil` | 110M | `load_splade` |
| MedCPT (query) | `ncbi/MedCPT-Query-Encoder` | 110M | `load_medcpt` |
| MedCPT (article) | `ncbi/MedCPT-Article-Encoder` | 110M | `load_medcpt` |

## Cross-encoders (reranking track)

| Role | HuggingFace ID | Params | Loader |
|---|---|---|---|
| BGE-reranker-base | `BAAI/bge-reranker-base` | 278M | `load_bge_reranker` |
| MedCPT cross-encoder | `ncbi/MedCPT-Cross-Encoder` | 110M | `load_medcpt_ce` |

## Generators (downstream MIRAGE / Section 5)

| Role | HuggingFace ID | Notes |
|---|---|---|
| flan-t5-base | `google/flan-t5-base` | Used for all four PubMedQA rows in `tables/mirage_accuracy.csv`. |
| Llama-3.2-3B | `meta-llama/Llama-3.2-3B-Instruct` | Gated; `huggingface-cli login` required. Not run for the numbers in the report (silently falls back to flan-t5-base). |

## Library versions

The pinned versions are in `requirements.txt`. The values that matter
most for retrieval reproducibility:

- `torch`
- `sentence-transformers`
- `transformers`
- `bm25s`
- `faiss-cpu`
- `lightgbm`
- `scikit-learn`
- `nltk`, `rouge-score`

`bm25s` in particular changed its tokenizer behaviour between the
versions used in Phase A and Phase B of this project (this is the
source of the NFCorpus BM25 calibration drift documented in
`main.tex` Section 4.3).

## Where the model loaders pick precision

`rageval.retrieval` picks `torch.float16` when CUDA is available, else
`torch.float32`. Free CUDA between loads:

```python
del model
free_cuda()
```

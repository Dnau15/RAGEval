# RAGEval

Experiments for **retrieval and RAG-style evaluation** on the [BEIR](https://github.com/beir-cellar/beir) benchmarks.
## Environment

**micromamba**

```bash
micromamba create -n rageval python=3.11 -c conda-forge -y
micromamba activate rageval
pip install torch sentence-transformers
```

The evaluation notebooks also use **BEIR**, **bm25s**, **FAISS**, **NLTK**, and related packages—install what each notebook imports (conda-forge is often easiest for FAISS on macOS).

**conda**

```bash
conda create -n rageval python=3.11 -y
conda activate rageval
pip install torch sentence-transformers
```

On **macOS**, prefer installing heavy native deps (for example **FAISS**, **PyTorch**) via **conda-forge** or the official PyTorch channel instead of only `pip`, so wheels match your OS and CPU/GPU. GPU-oriented FAISS builds are still mainly aimed at Linux; CPU FAISS is usually the practical option on Mac.

## Run

Activate the env, then smoke-test the stack (PyTorch + `sentence-transformers`):

```bash
bash test.sh
```

Open notebooks:

```bash
jupyter lab notebooks/
```

"""Generate the .tex table fragments used by main.tex from CSVs in tables/.

The CSVs in tables/ are a committed snapshot of the pipeline outputs that
live in notebooks/results/. main.tex pulls each fragment with
\\input{tables/<name>.tex}, so the report compiles from saved CSVs and the
numbers never get hand-typed.

Run:
    python scripts/build_tables.py              # rebuild .tex from tables/*.csv
    python scripts/build_tables.py --refresh    # pull fresh CSVs from
                                                # notebooks/results/ first

Each render_* function below maps one CSV to one tabular fragment. To add a
new table, write a CSV under tables/, add a render_* for it, and \\input it
from main.tex.
"""

import argparse
import shutil
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "tables"
FEEDBACK2 = ROOT / "notebooks" / "results" / "feedback2" / "tables"
NEXT_STAGE = ROOT / "notebooks" / "results" / "next_stage" / "tables"

NA = "{---}"  # what we print when a cell is NaN / blank


def fmt(x, d=4):
    if x is None or (isinstance(x, float) and pd.isna(x)) or x == "":
        return NA
    if isinstance(x, int):
        return f"{x:,}"
    if isinstance(x, float):
        return f"{x:.{d}f}"
    return str(x)


def fmt_int(x):
    if x is None or (isinstance(x, float) and pd.isna(x)) or x == "":
        return NA
    return f"{int(x):,}"


def fmt_signed(x, d=4):
    s = fmt(x, d)
    if s == NA:
        return s
    if isinstance(x, (int, float)) and x > 0:
        return f"+{x:.{d}f}"
    return s


def best_idx(series, mode="max"):
    """Index of the best value (max or min), ignoring NaN."""
    s = pd.to_numeric(series, errors="coerce")
    if s.notna().sum() == 0:
        return None
    return int(s.idxmax() if mode == "max" else s.idxmin())


def write(name, body):
    out = TABLES / f"{name}.tex"
    out.write_text(body)
    print(f"  wrote {out.relative_to(ROOT)}")


# -- one render_* per table ---------------------------------------------------

def render_datasets_manifest():
    df = pd.read_csv(TABLES / "datasets_manifest.csv")
    rows = []
    for _, r in df.iterrows():
        rows.append(
            f"{r['dataset']} & {fmt_int(r['n_docs'])} & "
            f"{fmt_int(r['n_queries_test'])} & {r['domain']} & "
            f"{r['query_style']} \\\\"
        )
    body = (
        "\\begin{tabular}{lrrll}\n"
        "\\toprule\n"
        "\\textbf{Dataset} & \\textbf{\\# Docs} & \\textbf{\\# Test Q} & "
        "\\textbf{Domain} & \\textbf{Query style} \\\\\n"
        "\\midrule\n"
        + "\n".join(rows) + "\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
    )
    write("datasets", body)


def render_retrievers_spec():
    df = pd.read_csv(TABLES / "retrievers_spec.csv")
    rows = []
    for _, r in df.iterrows():
        rows.append(
            f"{r['system']} & {r['type']} & {r['bias_class']} & "
            f"\\texttt{{{r['checkpoint']}}} \\\\"
        )
    body = (
        "\\begin{tabular}{llll}\n"
        "\\toprule\n"
        "\\textbf{System} & \\textbf{Type} & \\textbf{Bias class} & "
        "\\textbf{Model / parameters} \\\\\n"
        "\\midrule\n"
        + "\n".join(rows) + "\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
    )
    write("retrievers", body)


def render_nfcorpus_full_metrics():
    df = pd.read_csv(TABLES / "nfcorpus_full_metrics.csv")
    df = df.set_index("Method").reindex(
        ["BM25", "Dense", "BGE-small", "E5-small", "SPLADE", "MedCPT"])

    def row(label, key, dense_only=False):
        cells = []
        vals = df[key].tolist()
        bi = best_idx(pd.Series(vals))
        for i, v in enumerate(vals):
            s = fmt(v)
            if bi == i:
                s = f"\\textbf{{{s}}}"
            cells.append(s)
        return f"{label} & " + " & ".join(cells) + " \\\\"

    lines = [
        row("nDCG@1", "nDCG@1"),
        row("nDCG@3", "nDCG@3"),
        row("nDCG@5", "nDCG@5"),
        row("nDCG@10", "nDCG@10"),
        "\\midrule",
        row("MAP@10", "MAP@10"),
        row("Recall@10", "Recall@10"),
        row("P@10", "P@10"),
        "\\midrule",
    ]
    # Empirical risk: lower is better, so flip the bold rule.
    def row_min(label, key):
        vals = df[key].tolist()
        bi = best_idx(pd.Series(vals), mode="min")
        cells = []
        for i, v in enumerate(vals):
            s = fmt(v)
            if bi == i:
                s = f"\\textbf{{{s}}}"
            cells.append(s)
        return f"{label} & " + " & ".join(cells) + " \\\\"

    lines.append(row_min("Empirical risk $\\hat{R}$", "Empirical_risk"))
    # Loss variance is descriptive, not a "best at max" quantity; no bold.
    plain = " & ".join(fmt(v) for v in df["Loss_variance"].tolist())
    lines.append(f"Loss variance & {plain} \\\\")

    body = (
        "\\begin{tabular}{lSSSSSS}\n"
        "\\toprule\n"
        "\\textbf{Metric} & \\textbf{BM25} & \\textbf{Dense} & "
        "\\textbf{BGE-small} & \\textbf{E5-small} & \\textbf{SPLADE} & "
        "\\textbf{MedCPT} \\\\\n"
        "\\midrule\n"
        + "\n".join(lines) + "\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
    )
    write("nfcorpus_full_metrics", body)


def render_multi_dataset():
    df = pd.read_csv(TABLES / "multi_dataset_ndcg10_v2.csv")
    df = df.set_index("Dataset").reindex(
        ["NFCorpus", "BioASQ-subset", "TREC-COVID", "SciFact", "ArguAna"])
    cols = ["BM25 nDCG@10", "Dense nDCG@10", "BGE-small nDCG@10",
            "E5-small nDCG@10", "SPLADE nDCG@10", "MedCPT nDCG@10"]

    rows = []
    for ds, r in df.iterrows():
        vals = [r[c] for c in cols]
        bi = best_idx(pd.Series(vals))
        cells = []
        for i, v in enumerate(vals):
            s = fmt(v)
            if bi == i and s != NA:
                s = f"\\textbf{{{s}}}"
            cells.append(s)
        label = ds + "$^{\\dagger}$" if ds == "BioASQ-subset" else ds
        rows.append(f"{label} & " + " & ".join(cells) + " \\\\")

    body = (
        "\\begin{tabular}{lSSSSSS}\n"
        "\\toprule\n"
        "\\textbf{Dataset} & \\textbf{BM25} & \\textbf{Dense} & "
        "\\textbf{BGE-small} & \\textbf{E5-small} & \\textbf{SPLADE} & "
        "\\textbf{MedCPT} \\\\\n"
        "\\midrule\n"
        + "\n".join(rows) + "\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\n% $^{\\dagger}$ BioASQ-subset is a stress case, not biomedical "
        "evidence. See §\\ref{subsec:multi}.\n"
    )
    write("multi_dataset", body)


def render_reranker():
    df = pd.read_csv(TABLES / "reranker_ndcg10.csv")
    # Two groups: BGE-reranker rows (Status in {'done','degenerate-corpus'}
    # with RerankedSystem containing 'BGE-reranker'), then MedCPT-CE rows.
    bge = df[df["RerankedSystem"].str.contains("BGE-reranker")]
    mcp = df[df["RerankedSystem"].str.contains("MedCPT-CE")]

    def row(r):
        ds = r["Dataset"]
        if r.get("Status") == "degenerate-corpus":
            ds = ds + "$^{\\dagger}$"
        # Bold the reranked nDCG if Delta positive.
        rer = fmt(r["Reranked_nDCG10"])
        if r["Delta"] > 0:
            rer = f"\\textbf{{{rer}}}"
        return (
            f"{ds} & {r['FirstStage']} + {r['RerankedSystem'].split('+')[-1]} "
            f"& {fmt(r['FirstStage_nDCG10'])} & {rer} & "
            f"{fmt_signed(r['Delta'])} \\\\"
        )

    lines = [row(r) for _, r in bge.iterrows()]
    lines.append("\\midrule")
    lines.extend(row(r) for _, r in mcp.iterrows())

    body = (
        "\\begin{tabular}{llSSS}\n"
        "\\toprule\n"
        "\\textbf{Dataset} & \\textbf{First-stage} & "
        "\\textbf{First-stage nDCG@10} & \\textbf{Reranked nDCG@10} & "
        "\\textbf{$\\Delta$} \\\\\n"
        "\\midrule\n"
        + "\n".join(lines) + "\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\n% $^{\\dagger}$ degenerate corpus, see BioASQ caveat.\n"
    )
    write("reranker", body)


def render_paired_bootstrap():
    df = pd.read_csv(TABLES / "paired_bootstrap_summary.csv")
    rows = []
    for _, r in df.iterrows():
        ci_lo = r.get("ci_lo")
        ci_hi = r.get("ci_hi")
        if pd.isna(ci_lo) or pd.isna(ci_hi):
            ci = NA
        else:
            ci = f"{{[{fmt_signed(ci_lo)},\\, {fmt_signed(ci_hi)}]}}"
        p = r.get("p_a_gt_b")
        p_str = "n/a" if (pd.isna(p) or p == "") else f"{p*100:.1f}\\%"
        if r["scheme"].startswith("Point"):
            p_str = "---"
        contrast = r["contrast"].replace(" - ", " $-$ ")
        rows.append(
            f"{contrast} & {r['scheme']} & "
            f"{fmt_signed(r['mean_diff'])} & {ci} & {p_str} \\\\"
        )

    body = (
        "\\begin{tabular}{llSSc}\n"
        "\\toprule\n"
        "\\textbf{Contrast} & \\textbf{Scheme} & "
        "\\textbf{Mean $\\hat{\\Delta}$} & \\textbf{95\\% CI} & "
        "$P(A{>}B)$ \\\\\n"
        "\\midrule\n"
        + "\n".join(rows) + "\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
    )
    write("paired_bootstrap", body)


def render_router_test():
    df = pd.read_csv(TABLES / "router_test.csv")
    label_map = {
        "Always_BM25": "Always BM25",
        "Always_BGE_small": "Always BGE-small",
        "Static_Hybrid_alpha_0.40": "Static Hybrid (BM25 $+$ BGE-small, $\\alpha\\!=\\!0.40$)",
        "Always_BGE_small+BGE_reranker": "Always BGE-small $+$ BGE-reranker@100",
        "Logistic_router": "Logistic router",
        "LightGBM_router": "LightGBM router",
        "Oracle_router": "Oracle router (upper bound)",
    }
    # Find the best Always baseline so we can bold it.
    base = df[df["Status"] == "baseline"]
    best_base = base.loc[base["nDCG@10"].idxmax(), "Method"]

    rows = []
    for _, r in df.iterrows():
        score = fmt(r["nDCG@10"])
        if r["Method"] == best_base:
            score = f"\\textbf{{{score}}}"
        rows.append(
            f"{label_map.get(r['Method'], r['Method'])} & "
            f"{score} & {fmt_signed(r['DeltaVsBestAlways'])} \\\\"
        )
        if r["Method"] == "Always_BGE_small+BGE_reranker":
            rows.append("\\midrule")
        if r["Method"] == "LightGBM_router":
            rows.append("\\midrule")

    body = (
        "\\begin{tabular}{lSS}\n"
        "\\toprule\n"
        "\\textbf{Method} & \\textbf{nDCG@10} & "
        "\\textbf{$\\Delta$ vs best Always} \\\\\n"
        "\\midrule\n"
        + "\n".join(rows) + "\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
    )
    write("router_test", body)


def render_router_features():
    df = pd.read_csv(TABLES / "router_feature_importance.csv")
    df = df.sort_values("gain", ascending=False)
    if "share" not in df.columns:
        total = df["gain"].sum()
        df["share"] = df["gain"] / total
    rows = []
    for _, r in df.iterrows():
        rows.append(
            f"{r['feature'].replace('_', '\\_')} & "
            f"{r['gain']:,.1f} & {r['share']*100:.1f}\\% \\\\"
        )
    body = (
        "\\begin{tabular}{lSS}\n"
        "\\toprule\n"
        "\\textbf{Feature} & \\textbf{Gain} & \\textbf{Share of total} \\\\\n"
        "\\midrule\n"
        + "\n".join(rows) + "\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
    )
    write("router_features", body)


def render_efficiency():
    df = pd.read_csv(TABLES / "efficiency.csv")
    rows = []
    for _, r in df.iterrows():
        rows.append(
            f"{r['Retriever']} & {r['Index build (s)']:.1f} & "
            f"{r['Latency (ms/q)']:.1f} & "
            f"{fmt_int(r['Throughput (q/s)'])} & {r['Index memory']} \\\\"
        )
    body = (
        "\\begin{tabular}{lrrrl}\n"
        "\\toprule\n"
        "\\textbf{Retriever} & \\textbf{Index build (s)} & "
        "\\textbf{Latency (ms/q)} & \\textbf{Throughput (q/s)} & "
        "\\textbf{Index memory} \\\\\n"
        "\\midrule\n"
        + "\n".join(rows) + "\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
    )
    write("efficiency", body)


def render_n_required():
    df = pd.read_csv(TABLES / "n_required.csv")
    rows = []
    for _, r in df.iterrows():
        rows.append(
            f"{r['A']} $-$ {r['B']} & {fmt_signed(r['observed_gap'])} & "
            f"{fmt_int(r['hoeffding_n_required_95pct'])} \\\\"
        )
    body = (
        "\\begin{tabular}{lSS}\n"
        "\\toprule\n"
        "\\textbf{Pair} & \\textbf{Observed gap $\\hat{\\Delta}$} & "
        "\\textbf{$n_{\\min}$} \\\\\n"
        "\\midrule\n"
        + "\n".join(rows) + "\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
    )
    write("n_required", body)


def render_subsampling():
    df = pd.read_csv(TABLES / "query_subset_resampling_summary.csv")
    name_map = {
        "Dense_minus_BM25": "Dense $-$ BM25",
        "Hybrid_minus_Dense": "Hybrid $-$ Dense",
    }
    order = ["BM25", "Dense", "Hybrid", "Dense_minus_BM25", "Hybrid_minus_Dense"]
    df = df.set_index("method").reindex(order)
    rows = []
    for m, r in df.iterrows():
        label = name_map.get(m, m)
        line = (
            f"{label} & {fmt(r['mean_subset_nDCG@10'])} & "
            f"{fmt(r['q025'])} & {fmt(r['q975'])} & "
            f"{fmt(r['std_subset_nDCG@10'])} \\\\"
        )
        rows.append(line)
        if m == "Hybrid":
            rows.append("\\midrule")
    body = (
        "\\begin{tabular}{lSSSS}\n"
        "\\toprule\n"
        "\\textbf{Quantity} & \\textbf{Mean} & "
        "\\textbf{$Q_{2.5}$} & \\textbf{$Q_{97.5}$} & \\textbf{Std} \\\\\n"
        "\\midrule\n"
        + "\n".join(rows) + "\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
    )
    write("subsampling", body)


def render_regression():
    df = pd.read_csv(TABLES / "delta_regression_coefficients.csv")
    rows = []
    for _, r in df.iterrows():
        coef = fmt_signed(r["Coef"])
        ci = f"{{[{fmt_signed(r['CI_lo'])},\\, {fmt_signed(r['CI_hi'])}]}}"
        p_val = float(r["p_boot"])
        p_str = "{$<\\!0.001$}" if p_val < 0.001 else f"{p_val:.3f}"
        sig = r["Sig"] if isinstance(r["Sig"], str) and r["Sig"] else "{---}"
        if sig != "{---}":
            sig = f"${sig}$"
        rows.append(
            f"{r['Feature']} & {coef} & {ci} & {p_str} & {sig} \\\\"
        )
    body = (
        "\\begin{tabular}{lSSSc}\n"
        "\\toprule\n"
        "\\textbf{Feature} & \\textbf{Coef} & \\textbf{95\\% CI} & "
        "\\textbf{Boot $p$} & \\textbf{Sig.} \\\\\n"
        "\\midrule\n"
        + "\n".join(rows) + "\n"
        "\\midrule\n"
        "$R^2$ & 0.071 & & & \\\\\n"
        "$n$ & 323 & & & \\\\\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
    )
    write("regression", body)


def _render_stratification(csv_name, out_name):
    df = pd.read_csv(TABLES / csv_name)
    rows = []
    cols = ["BM25_nDCG@10", "Dense_nDCG@10", "Hybrid_nDCG@10"]
    for _, r in df.iterrows():
        # Bold the best of BM25/Dense/Hybrid per row.
        vals = [r[c] for c in cols]
        bi = best_idx(pd.Series(vals))
        cells = []
        for i, v in enumerate(vals):
            s = fmt(v)
            if i == bi:
                s = f"\\textbf{{{s}}}"
            cells.append(s)
        rows.append(
            f"{r['stratum']} & {fmt_int(r['n_queries'])} & " +
            " & ".join(cells) + " & " +
            fmt_signed(r["Dense_minus_BM25"]) + " \\\\"
        )
    body = (
        "\\begin{tabular}{lSSSSS}\n"
        "\\toprule\n"
        "\\textbf{Stratum} & \\textbf{$n$} & \\textbf{BM25} & "
        "\\textbf{Dense} & \\textbf{Hybrid} & \\textbf{Dense$-$BM25} \\\\\n"
        "\\midrule\n"
        + "\n".join(rows) + "\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
    )
    write(out_name, body)


def render_vocab_gap_strat():
    _render_stratification(
        "vocabulary_gap_stratification_ndcg10.csv", "vocab_gap_strat")


def render_tech_strat():
    _render_stratification(
        "technicality_stratification_ndcg10.csv", "tech_strat")


def render_train_test_gap():
    df = pd.read_csv(TABLES / "train_test_gap_summary.csv")
    fs = df[df["kind"] == "first-stage"]
    rt = df[df["kind"] == "router"]
    labels = list(fs["model"]) + list(rt["model"])
    # Build header dynamically.
    header_cells = []
    for m in labels:
        nice = m.replace("_router", "").replace("_", " ")
        header_cells.append(f"\\textbf{{{nice}}}")
    header = "\\textbf{Quantity} & " + " & ".join(header_cells) + " \\\\"

    def emit(label, key, formatter=fmt):
        cells = []
        for m in labels:
            r = df[df["model"] == m].iloc[0]
            v = r.get(key)
            cells.append(formatter(v) if not pd.isna(v) else NA)
        return f"{label} & " + " & ".join(cells) + " \\\\"

    rows = [
        emit("$R_{\\text{train}}$", "R_train"),
        emit("$R_{\\text{test}}$", "R_test"),
    ]
    # Gap row: bold the LightGBM cell because the report calls it out.
    cells = []
    for m in labels:
        r = df[df["model"] == m].iloc[0]
        s = fmt_signed(r["gap"])
        if m == "LightGBM_router":
            s = f"\\textbf{{{s}}}"
        cells.append(s)
    rows.append("Gap $\\Delta_{\\text{gen}}$ & " + " & ".join(cells) + " \\\\")
    # Mann-Whitney row: only first-stage cells have a value.
    mw = []
    for m in labels:
        r = df[df["model"] == m].iloc[0]
        v = r.get("mw_p")
        mw.append(fmt(v) if not (pd.isna(v) or v == "") else NA)
    rows.append("Mann--Whitney $p$ & " + " & ".join(mw) + " \\\\")

    col_spec = "l" + "S" * len(labels)
    body = (
        f"\\begin{{tabular}}{{{col_spec}}}\n"
        "\\toprule\n"
        f"{header}\n"
        "\\midrule\n"
        + "\n".join(rows) + "\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
    )
    write("train_test_gap", body)


def render_reranker_depth():
    """Wide table: rows = (dataset, first-stage); columns = depths k.
    Cells show Delta = reranked - first-stage."""
    df = pd.read_csv(TABLES / "reranker_depth_sweep.csv")
    depths = sorted(df["k"].unique().tolist())
    combos = df[["Dataset", "FirstStage"]].drop_duplicates().values.tolist()

    rows = []
    for ds, fs in combos:
        cells = []
        for k in depths:
            row = df[(df["Dataset"] == ds) & (df["FirstStage"] == fs)
                     & (df["k"] == k)]
            if row.empty or pd.isna(row.iloc[0]["Delta"]):
                cells.append(NA)
            else:
                cells.append(fmt_signed(row.iloc[0]["Delta"]))
        ds_label = ds + "$^{\\dagger}$" if ds == "BioASQ-subset" else ds
        rows.append(f"{ds_label} & {fs} & " + " & ".join(cells) + " \\\\")

    header_cells = " & ".join(f"\\textbf{{$k={k}$}}" for k in depths)
    col_spec = "ll" + "S" * len(depths)
    body = (
        f"\\begin{{tabular}}{{{col_spec}}}\n"
        "\\toprule\n"
        f"\\textbf{{Dataset}} & \\textbf{{First-stage}} & {header_cells} \\\\\n"
        "\\midrule\n"
        + "\n".join(rows) + "\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\n% Cells show $\\Delta$ nDCG@10 = reranked $-$ first-stage. "
        "Empty cells await the Colab rerun.\n"
        "% $^{\\dagger}$ degenerate corpus, see BioASQ caveat.\n"
    )
    write("reranker_depth", body)


def render_reranker_ci():
    """Tall table: paired bootstrap CI at k=100 for each (dataset, FS)."""
    df = pd.read_csv(TABLES / "reranker_paired_bootstrap.csv")
    df = df[df["k"] == 100].copy()
    rows = []
    for _, r in df.iterrows():
        ds = r["Dataset"]
        if ds == "BioASQ-subset":
            ds = ds + "$^{\\dagger}$"
        ci_lo = r.get("ci_lo")
        ci_hi = r.get("ci_hi")
        if pd.isna(ci_lo) or pd.isna(ci_hi):
            ci = NA
            p = NA
        else:
            ci = f"{{[{fmt_signed(ci_lo)},\\, {fmt_signed(ci_hi)}]}}"
            p_val = r.get("p_a_gt_b")
            p = NA if pd.isna(p_val) else f"{p_val*100:.1f}\\%"
        rows.append(
            f"{ds} & {r['FirstStage']} & {fmt_int(r['n_queries'])} & "
            f"{fmt_signed(r['mean_diff'])} & {ci} & {p} \\\\"
        )
    body = (
        "\\begin{tabular}{llSSSc}\n"
        "\\toprule\n"
        "\\textbf{Dataset} & \\textbf{First-stage} & \\textbf{$n$} & "
        "\\textbf{$\\hat{\\Delta}$} & \\textbf{95\\% CI} & "
        "\\textbf{$P(\\text{rerank}{>}\\text{FS})$} \\\\\n"
        "\\midrule\n"
        + "\n".join(rows) + "\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\n% Paired bootstrap, $B = 10{,}000$, at the original $k=100$ "
        "candidate depth. CI and $P$ columns await the Colab rerun.\n"
        "% $^{\\dagger}$ degenerate corpus, see BioASQ caveat.\n"
    )
    write("reranker_ci", body)


def render_mirage():
    df = pd.read_csv(TABLES / "mirage_accuracy.csv")
    rows = []
    for _, r in df.iterrows():
        if r["Status"] == "not-run":
            continue
        acc = fmt(r["Accuracy"], d=3)
        if r["Status"] == "saturated":
            acc = f"\\textbf{{{acc}}}"
        rows.append(
            f"{r['Task']} & {r['Retriever']} & {r['Generator']} & {acc} \\\\"
        )
    rows.append("\\midrule")
    rows.append(
        "\\multicolumn{4}{l}{\\emph{Llama-3.2-3B-Instruct rows: not run "
        "(gated; no HF login in this Colab session).}} \\\\"
    )
    rows.append(
        "\\multicolumn{4}{l}{\\emph{BioASQ Y/N rows: not run "
        "(loader returned no yes/no records).}} \\\\"
    )
    body = (
        "\\begin{tabular}{lllS}\n"
        "\\toprule\n"
        "\\textbf{Task} & \\textbf{Retriever} & \\textbf{Generator} & "
        "\\textbf{Accuracy} \\\\\n"
        "\\midrule\n"
        + "\n".join(rows) + "\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
    )
    write("mirage", body)


# -- entry point --------------------------------------------------------------

RENDERS = [
    # Configuration tables (datasets, retrievers spec) stay manual in main.tex:
    # they carry LaTeX math (k_1=1.5) and grouping braces that don't pass
    # cleanly through a generic CSV renderer. The CSVs are kept as a manifest.
    render_nfcorpus_full_metrics,
    render_multi_dataset,
    render_reranker,
    render_reranker_depth,
    render_reranker_ci,
    render_paired_bootstrap,
    render_router_test,
    render_router_features,
    render_efficiency,
    render_n_required,
    render_subsampling,
    render_regression,
    render_vocab_gap_strat,
    render_tech_strat,
    render_train_test_gap,
    render_mirage,
]


def refresh_from_pipeline():
    """Copy fresh CSVs from notebooks/results/ into tables/, overwriting."""
    copied = 0
    for src_dir in (FEEDBACK2, NEXT_STAGE):
        if not src_dir.exists():
            continue
        for csv in src_dir.glob("*.csv"):
            dst = TABLES / csv.name
            shutil.copy2(csv, dst)
            print(f"  pulled {csv.relative_to(ROOT)} -> tables/")
            copied += 1
    if not copied:
        print("  (no CSVs found in notebooks/results/, using committed snapshot)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--refresh", action="store_true",
        help="Copy fresh CSVs from notebooks/results/ into tables/ first.",
    )
    args = p.parse_args()

    if args.refresh:
        print("Refreshing tables/ from pipeline outputs ...")
        refresh_from_pipeline()

    print(f"Rendering .tex fragments into {TABLES.relative_to(ROOT)}/ ...")
    for fn in RENDERS:
        fn()
    print(f"\ndone. {len(RENDERS)} fragments rendered.")


if __name__ == "__main__":
    sys.exit(main())

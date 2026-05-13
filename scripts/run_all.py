"""Run the full pipeline in dependency order.

    1. analysis.py   -- writes hybrid_test_comparison.csv
                        (read by phase_a.write_n_required)
    2. phase_a.py    -- first-stage + multi-dataset + n_required + efficiency
    3. phase_b.py    -- BioASQ + reranker + router + MIRAGE

A full run on a Colab T4 is roughly 2-3 hours.
"""

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))


def run(label, fn):
    print("\n" + "=" * 70)
    print(f"  {label}")
    print("=" * 70)
    t0 = time.time()
    fn()
    print(f"\n  done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    import analysis
    import phase_a
    import phase_b

    def analysis_stage():
        alpha = analysis.split_metrics_and_alpha()
        analysis.boot_and_subsampling(alpha)
        analysis.regression_and_stratifications()

    def phase_a_stage():
        phase_a.first_stage_and_multi_dataset()
        phase_a.write_n_required()
        phase_a.write_efficiency()

    def phase_b_stage():
        phase_b.build_bioasq_subset()
        phase_b.run_rerankers()
        phase_b.rerank_depth_and_ci()
        phase_b.run_router()
        phase_b.run_mirage()

    run("analysis.py: paired bootstrap, regression, stratifications", analysis_stage)
    run("phase_a.py: first-stage + multi-dataset + n_required + efficiency", phase_a_stage)
    run("phase_b.py: BioASQ + reranker + router + MIRAGE", phase_b_stage)

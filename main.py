import sys
import time
import argparse
from pathlib import Path

import pm4py
import torch

from utils import *
from petri_net_ops import infer_final_marking, visible_labels
from repair_utils import sequential_lpmr_repair_improved

import warnings
warnings.filterwarnings("ignore", category=UserWarning)


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--max-traces", type=int, default=0)
    parser.add_argument("--min-support", type=int, default=20)
    parser.add_argument("--min-relative-support", type=float, default=0.005)
    parser.add_argument("--relation-candidate-limit", type=int, default=80)
    parser.add_argument("--fragment-candidate-limit", type=int, default=80)
    parser.add_argument("--max-insert-len", type=int, default=3)
    parser.add_argument("--max-repairs", type=int, default=2)
    parser.add_argument("--max-validation-trials", type=int, default=12)
    parser.add_argument("--sample-metrics-traces", type=int, default=450)
    parser.add_argument("--sample-variant-coverage", type=int, default=120)
    parser.add_argument("--sample-validation-trials", type=int, default=60)
    parser.add_argument("--sample-candidate-keep", type=int, default=14)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-steps", type=int, default=6)
    parser.add_argument("--action-candidate-limit", type=int, default=16)
    parser.add_argument("--precision-validation-trials", type=int, default=8)
    parser.add_argument("--full-validation-top-k", type=int, default=2)
    parser.add_argument("--decision-metrics-traces", type=int, default=450)
    parser.add_argument("--no-improvement-patience", type=int, default=5)
    parser.add_argument("--rollback-to-best", action='store_true')
    parser.add_argument("--ppo-lr", type=float, default=1e-4)
    parser.add_argument("--ppo-epochs", type=int, default=12)
    parser.add_argument("--ppo-clip", type=float, default=0.2)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--reward-guided-selection-weight", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=2.0)
    parser.add_argument("--negative-reward-penalty", type=float, default=0.05)
    parser.add_argument("--progress-bonus", type=float, default=0.01)
    parser.add_argument("--catastrophic-loss", type=float, default=0.10)
    parser.add_argument("--catastrophic-penalty", type=float, default=0.50)
    parser.add_argument("--action-complexity-penalty", type=float, default=0.0002)
    parser.add_argument("--action-duplicate-penalty", type=float, default=0.002)
    parser.add_argument("--fitness-near-one", type=float, default=0.995)
    parser.add_argument("--max-fitness-loss", type=float, default=0.002)
    parser.add_argument("--max-precision-loss", type=float, default=0.03)
    parser.add_argument("--max-simplicity-loss", type=float, default=0.04)
    parser.add_argument("--metric-tolerance", type=float, default=1e-9)
    parser.add_argument("--alignment-sample-traces", type=int, default=250)
    parser.add_argument("--alignment-candidate-limit", type=int, default=80)
    parser.add_argument("--relaxation-candidate-limit", type=int, default=40)
    parser.add_argument("--prune-candidate-limit", type=int, default=30)
    parser.add_argument("--buffer-capacity", type=int, default=32)
    parser.add_argument("--force-explore-steps", type=int, default=3)
    parser.add_argument("--stop-patience", type=int, default=2)
    parser.add_argument("--accept-reward-threshold", type=float, default=0.0)
    parser.add_argument("--candidate-max-retry", type=int, default=2)
    parser.add_argument("--gamma", type=float, default=0.97)
    parser.add_argument("--gae-lambda", type=float, default=0.92)
    parser.add_argument("--clear-rollout-after-update", action="store_true", default=True)
    parser.add_argument("--deterministic-policy", action="store_true")
    parser.add_argument("--greedy-accepted-action", type=lambda x: str(x).lower() not in {"0", "false", "no"}, default=True)
    parser.add_argument("--mask-unacceptable-actions", action="store_true", default=True)
    parser.add_argument("--stop-reward", type=float, default=-0.002)
    parser.add_argument("--invalid-action-penalty", type=float, default=0.05)
    parser.add_argument("--unacceptable-action-penalty", type=float, default=0.025)
    parser.add_argument("--fitness-floor-penalty", type=float, default=8.0)
    parser.add_argument("--fitness-floor-bonus", type=float, default=0.01)
    parser.add_argument("--reward-composite-weight", type=float, default=1.0)
    parser.add_argument("--reward-fitness-weight", type=float, default=1.2)
    parser.add_argument("--reward-precision-weight", type=float, default=0.7)
    parser.add_argument("--reward-f1-weight", type=float, default=0.8)
    parser.add_argument("--reward-simplicity-weight", type=float, default=0.1)
    parser.add_argument("--accept-composite-epsilon", type=float, default=1e-7)
    parser.add_argument("--accept-f1-epsilon", type=float, default=1e-7)
    parser.add_argument("--accept-precision-epsilon", type=float, default=1e-7)
    parser.add_argument("--accept-fitness-epsilon", type=float, default=1e-7)
    parser.add_argument("--max-f1-loss", type=float, default=0.002)
    parser.add_argument("--max-composite-loss", type=float, default=0.0005)
    parser.add_argument("--fallback-min-support", type=int, default=5)
    parser.add_argument("--fallback-min-relative-support", type=float, default=0.0005)
    parser.add_argument("--fallback-action-candidate-limit", type=int, default=48)
    parser.add_argument("--fallback-relation-candidate-limit", type=int, default=160)
    parser.add_argument("--fallback-fragment-candidate-limit", type=int, default=160)
    parser.add_argument("--fallback-alignment-candidate-limit", type=int, default=160)
    parser.add_argument("--fallback-prune-candidate-limit", type=int, default=80)
    parser.add_argument("--fallback-max-sample-precision-loss", type=float, default=0.005)
    parser.add_argument("--fallback-max-sample-f1-loss", type=float, default=0.01)
    parser.add_argument("--fallback-hard-sample-precision-loss", type=float, default=0.03)
    parser.add_argument("--fallback-hard-sample-f1-loss", type=float, default=0.04)
    parser.add_argument("--fallback-max-ranked-fitness-gain", type=float, default=0.02)
    parser.add_argument("--full-fallback-trials", type=int, default=4)
    parser.add_argument("--fallback-sample-validation-trials", type=int, default=10)
    parser.add_argument("--fallback-relation-variant-factor", type=int, default=2)
    parser.add_argument("--fallback-early-stop-composite-gain", type=float, default=0.001)
    parser.add_argument("--final-max-composite-loss", type=float, default=0.0)
    parser.add_argument("--use-original-cache", type=lambda x: str(x).lower() not in {"0", "false", "no"}, default=False)
    parser.add_argument("--candidate-eval-mode", choices=["fast", "full"], default="fast")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    root = Path.cwd()
    data_dir = root / args.data_dir
    xes_path = data_dir / args.data / f"{args.data}.xes"
    pnml_path = data_dir / args.data / f"{args.data}_petrinet.pnml"
    if not xes_path.exists() or not pnml_path.exists():
        raise FileNotFoundError("Missing input files")

    base_net, initial_marking, final_marking = pm4py.read_pnml(str(pnml_path))
    final_marking = infer_final_marking(base_net, final_marking)
    model_labels = visible_labels(base_net)

    variants, trace_count, event_count = read_xes_variants_streaming(xes_path, model_labels, args.max_traces)
    full_log = build_event_log_from_variants(variants, 0)
    sample_log = build_sample_log_from_variants(variants, args.sample_metrics_traces, args.sample_variant_coverage)
    alignment_log = build_sample_log_from_variants(variants, args.alignment_sample_traces, args.sample_variant_coverage)

    cache_report = root / "output" / f"{args.data}_RL_repair_report.md"
    original_metrics = load_original_metrics_cache(cache_report) if args.use_original_cache else None
    if original_metrics is None:
        original_metrics = compute_quality_metrics(full_log, base_net, initial_marking, final_marking)
    decision_log = full_log if args.decision_metrics_traces == 0 else build_sample_log_from_variants(
        variants, args.decision_metrics_traces, args.sample_variant_coverage)

    repaired_net, repaired_im, repaired_fm, repaired_metrics, selected, decisions, \
        rem_t, rem_p, rem_a, last_batch = sequential_lpmr_repair_improved(
            base_net, initial_marking, final_marking, variants, trace_count,
            full_log, decision_log, alignment_log, original_metrics, args)

    out_dir = root / "output_repair"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_pnml = out_dir / f"{args.data}_repaired_model.pnml"
    out_report = out_dir / f"{args.data}_repair_report.md"

    export_pnml(repaired_net, repaired_im, repaired_fm, out_pnml)
    write_report(out_report, args, trace_count, event_count, len(variants),
                 last_batch.relation_candidates, last_batch.fragment_candidates,
                 last_batch.alignment_candidates, last_batch.relaxation_candidates,
                 last_batch.prune_candidates,
                 last_batch.candidates, selected, decisions,
                 original_metrics, repaired_metrics, out_pnml)

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
import argparse
import os
from pathlib import Path

from pcft.cli.train import train_id_dr_stateft
from pcft.evaluate import DATASETS, evaluate_adapter


def build_parser():
    parser = argparse.ArgumentParser(description="Train and evaluate Llama 3B ID-DR StateFT")
    parser.add_argument("--base-model", default="meta-llama/Llama-3.2-3B")
    parser.add_argument("--data-path", default="datasets/commonsense_170k.json")
    parser.add_argument("--test-data-path", default="datasets")
    parser.add_argument("--output-dir", default="checkpoints/llama-3.2-3b-id-dr-stateft")
    parser.add_argument("--results-file")
    parser.add_argument(
        "--allocation-method", choices=["fixed", "loss_exchange", "id_exchange"],
        default="id_exchange",
    )
    parser.add_argument("--initial-rank", type=int, default=64)
    parser.add_argument("--rank-min", type=int, default=8)
    parser.add_argument("--rank-max", type=int, default=128)
    parser.add_argument("--rank-quantum", type=int, default=8)
    parser.add_argument("--global-rank-budget", type=int)
    parser.add_argument("--alpha", type=float, default=16.0)
    parser.add_argument(
        "--scaling-mode",
        choices=["alpha", "alpha_over_sqrt_rank", "alpha_over_rank"],
        default="alpha_over_sqrt_rank",
    )
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--micro-batch-size", type=int, default=4)
    parser.add_argument("--num-epochs", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--cutoff-len", type=int, default=256)
    parser.add_argument("--rank-calibration-size", type=int, default=1000)
    parser.add_argument("--validation-size", type=int, default=2000)
    parser.add_argument("--eval-steps", type=int, default=200)
    parser.add_argument("--save-steps", type=int, default=200)
    parser.add_argument("--allocation-interval", type=int, default=600)
    parser.add_argument("--allocation-warmup-events", type=int, default=3)
    parser.add_argument("--rank-freeze-ratio", type=float, default=0.2)
    parser.add_argument("--id-sample-size", type=int, default=256)
    parser.add_argument("--id-bootstrap-repeats", type=int, default=20)
    parser.add_argument("--id-bootstrap-fraction", type=float, default=0.8)
    parser.add_argument("--id-uncertainty-lambda", type=float, default=1.0)
    parser.add_argument("--id-ema-alpha", type=float, default=0.3)
    parser.add_argument("--id-prior-beta", type=float, default=2.0)
    parser.add_argument("--id-regularization-tau", type=float, default=0.05)
    parser.add_argument("--rank-switch-cost", type=float, default=0.001)
    parser.add_argument("--move-threshold", type=float, default=0.0)
    parser.add_argument("--rank-probe-size", type=int, default=128)
    parser.add_argument("--direct-verify-size", type=int, default=256)
    parser.add_argument("--receiver-count", type=int, default=10)
    parser.add_argument("--donor-count", type=int, default=10)
    parser.add_argument("--direct-verify-pairs", type=int, default=6)
    parser.add_argument("--max-transfers-per-event", type=int, default=4)
    parser.add_argument("--rank-exploration-probability", type=float, default=0.15)
    parser.add_argument("--exploration-transfers", type=int, default=1)
    parser.add_argument("--rank-transition-steps", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--datasets", nargs="+", default=DATASETS)
    parser.add_argument("--length-norm", choices=["none", "avg"], default="none")
    parser.add_argument("--max-eval-examples", type=int)
    parser.add_argument("--resume-from-checkpoint")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--evaluate-supernet", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def resolve(path, root):
    path = Path(path).expanduser()
    return str(path.resolve() if path.is_absolute() else (root / path).resolve())


def train_kwargs(args):
    names = [
        "base_model", "data_path", "output_dir", "allocation_method", "initial_rank",
        "rank_min", "rank_max", "rank_quantum", "global_rank_budget", "alpha",
        "scaling_mode", "dropout", "batch_size", "micro_batch_size", "num_epochs",
        "max_steps", "learning_rate", "cutoff_len", "rank_calibration_size",
        "validation_size", "eval_steps", "save_steps", "allocation_interval",
        "allocation_warmup_events", "rank_freeze_ratio", "id_sample_size",
        "id_bootstrap_repeats", "id_bootstrap_fraction", "id_uncertainty_lambda",
        "id_ema_alpha", "id_prior_beta", "id_regularization_tau", "rank_switch_cost",
        "move_threshold", "rank_probe_size", "direct_verify_size", "receiver_count",
        "donor_count", "direct_verify_pairs", "max_transfers_per_event",
        "rank_exploration_probability", "exploration_transfers", "rank_transition_steps",
        "resume_from_checkpoint", "seed",
    ]
    return {name: getattr(args, name) for name in names}


def main():
    args = build_parser().parse_args()
    root = Path(__file__).resolve().parent
    os.chdir(root)
    for name in ("data_path", "test_data_path", "output_dir", "results_file"):
        value = getattr(args, name)
        if value:
            setattr(args, name, resolve(value, root))
    if "8b" in args.base_model.lower():
        raise SystemExit("Error: this repository is 3B-only")
    default_budget = 56 * args.initial_rank
    budget = args.global_rank_budget or default_budget
    if budget != default_budget:
        raise SystemExit(
            f"Initial uniform ranks require --global-rank-budget {default_budget}, got {budget}"
        )
    print(f"Base model:  {args.base_model}")
    print(f"Adapter:     {args.output_dir}")
    print(f"Allocation:  {args.allocation_method}")
    print(
        f"Rank:        min={args.rank_min}, initial={args.initial_rank}, "
        f"max={args.rank_max}, quantum={args.rank_quantum}, budget={budget}"
    )
    if args.dry_run:
        print("Dry run passed")
        return

    result = None
    if not args.skip_train:
        if not Path(args.data_path).is_file():
            raise SystemExit(f"Training data not found: {args.data_path}")
        result = train_id_dr_stateft(**train_kwargs(args))
    if not args.skip_eval:
        adapter_dir = args.output_dir
        if result and not args.evaluate_supernet:
            adapter_dir = result["compact_dir"]
        elif not args.evaluate_supernet and (Path(args.output_dir) / "compact").is_dir():
            adapter_dir = str(Path(args.output_dir) / "compact")
        evaluate_adapter(
            base_model=args.base_model,
            adapter_dir=adapter_dir,
            test_data_path=args.test_data_path,
            datasets=args.datasets,
            results_file=args.results_file or str(Path(args.output_dir) / "evaluation.csv"),
            length_norm=args.length_norm,
            max_examples=args.max_eval_examples,
        )


if __name__ == "__main__":
    main()

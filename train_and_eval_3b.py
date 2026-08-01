#!/usr/bin/env python3
import argparse
import os
from pathlib import Path

from pcft.cli.train import train_double_control
from pcft.evaluate import DATASETS, evaluate_adapter


def parse_args():
    parser = argparse.ArgumentParser(description="Train and evaluate 3B Double State-FT")
    parser.add_argument("--base-model", default="meta-llama/Llama-3.2-3B")
    parser.add_argument("--data-path", default="datasets/commonsense_170k.json")
    parser.add_argument("--test-data-path", default="datasets")
    parser.add_argument("--output-dir", default="checkpoints/llama-3.2-3b-double")
    parser.add_argument("--results-file", default=None)
    parser.add_argument("--initial-rank", type=int, default=64)
    parser.add_argument("--rank-min", type=int, default=8)
    parser.add_argument("--rank-max", type=int, default=128)
    parser.add_argument("--alpha", type=float, default=16.0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--micro-batch-size", type=int, default=4)
    parser.add_argument("--num-epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--cutoff-len", type=int, default=256)
    parser.add_argument("--val-set-size", type=int, default=2000)
    parser.add_argument("--eval-steps", type=int, default=200)
    parser.add_argument("--save-steps", type=int, default=200)
    parser.add_argument("--id-sample-size", type=int, default=256)
    parser.add_argument("--warmup-evals", type=int, default=3)
    parser.add_argument("--datasets", nargs="+", default=DATASETS)
    parser.add_argument("--length-norm", choices=["none", "avg"], default="none")
    parser.add_argument("--resume-from-checkpoint", default=None)
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def resolve(path, root):
    path = Path(path).expanduser()
    return str(path.resolve() if path.is_absolute() else (root / path).resolve())


def main():
    args = parse_args()
    root = Path(__file__).resolve().parent
    os.chdir(root)
    args.data_path = resolve(args.data_path, root)
    args.test_data_path = resolve(args.test_data_path, root)
    args.output_dir = resolve(args.output_dir, root)
    if args.results_file:
        args.results_file = resolve(args.results_file, root)

    if "8b" in args.base_model.lower():
        raise SystemExit("Error: this project is 3B-only")
    print(f"Base model: {args.base_model}")
    print(f"Adapter:    {args.output_dir}")
    print(
        f"Rank:       min={args.rank_min}, initial={args.initial_rank}, max={args.rank_max}"
    )
    if args.dry_run:
        print("Dry run passed")
        return

    if not args.skip_train:
        if not Path(args.data_path).is_file():
            raise SystemExit(f"Training data not found: {args.data_path}")
        train_double_control(
            base_model=args.base_model,
            data_path=args.data_path,
            output_dir=args.output_dir,
            initial_rank=args.initial_rank,
            rank_min=args.rank_min,
            rank_max=args.rank_max,
            alpha=args.alpha,
            batch_size=args.batch_size,
            micro_batch_size=args.micro_batch_size,
            num_epochs=args.num_epochs,
            learning_rate=args.learning_rate,
            cutoff_len=args.cutoff_len,
            val_set_size=args.val_set_size,
            eval_steps=args.eval_steps,
            save_steps=args.save_steps,
            id_sample_size=args.id_sample_size,
            warmup_evals=args.warmup_evals,
            resume_from_checkpoint=args.resume_from_checkpoint,
        )

    if not args.skip_eval:
        evaluate_adapter(
            base_model=args.base_model,
            adapter_dir=args.output_dir,
            test_data_path=args.test_data_path,
            datasets=args.datasets,
            results_file=args.results_file,
            length_norm=args.length_norm,
        )


if __name__ == "__main__":
    main()

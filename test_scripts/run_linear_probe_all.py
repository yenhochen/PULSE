"""Run all linear-probe configs through train + eval for a small epoch budget.

Use this to verify the full pretrain pipeline (including downstream eval) before
launching full paper reproduction runs.

Example:
  export PYTHONPATH=$(pwd)
  python test_scripts/run_linear_probe_all.py --epochs 2 --seeds 0 --run
"""

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

from utils.experiment_commands import (
    build_linear_probe_commands,
    write_command_script,
)

DEFAULT_SAVE_ROOT = "experiments/linear_probe_check"
DEFAULT_COMMANDS_PATH = "test_scripts/commands_linear_probe_check.sh"


def get_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Generate and optionally run all linear-probe configs for a short "
            "epoch budget (full train + eval pipeline)."
        )
    )
    parser.add_argument(
        "-td",
        "--top_config_dir",
        default="configs/linear_probe",
        help="Root directory of linear-probe yaml configs",
    )
    parser.add_argument(
        "-e",
        "--epochs",
        type=int,
        default=2,
        help="Training epochs (rebar cross-attn stage capped too if included)",
    )
    parser.add_argument(
        "--seeds",
        default="0",
        help="Comma-separated seeds (default: 0)",
    )
    parser.add_argument(
        "--save_root",
        default=DEFAULT_SAVE_ROOT,
        help=f"Output root (default: {DEFAULT_SAVE_ROOT})",
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="Only run configs for one dataset (e.g. har, ecg, ppg, eeg_2ch)",
    )
    parser.add_argument(
        "-o",
        "--output_path",
        default=DEFAULT_COMMANDS_PATH,
        help=f"Generated shell script path (default: {DEFAULT_COMMANDS_PATH})",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Run even when scores.pkl already exists under save_root",
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Execute the generated command script after writing it",
    )
    parser.add_argument(
        "--include-rebar",
        action="store_true",
        help="Include rebar configs (skipped by default; stage-1 cross-attn is slow)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without writing a script or running",
    )
    return parser


def main():
    args = get_parser().parse_args()
    seeds = tuple(int(s.strip()) for s in args.seeds.split(",") if s.strip())

    commands = build_linear_probe_commands(
        top_config_dir=args.top_config_dir,
        seeds=seeds,
        epochs=args.epochs,
        save_root=args.save_root,
        force=args.force,
        dataset=args.dataset,
        exclude_models=() if args.include_rebar else ("rebar",),
    )

    if args.dry_run:
        for cmd in commands:
            print(cmd)
        print(f"\n{len(commands)} command(s)")
        return

    if not commands:
        print("No commands to run.")
        return

    output_path = write_command_script(commands, args.output_path)
    print(f"Wrote {len(commands)} commands to {output_path}")

    if args.run:
        print(f"Running {output_path} ...")
        result = subprocess.run(["bash", output_path], cwd=REPO_ROOT)
        if result.returncode != 0:
            sys.exit(result.returncode)


if __name__ == "__main__":
    main()

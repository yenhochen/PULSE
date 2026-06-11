"""Generate scripts/run/pretrain.py commands for linear-probe configs."""

import argparse

from utils.experiment_commands import (
    DEFAULT_SEEDS,
    GENERATED_DIR,
    build_linear_probe_commands,
    write_command_script,
)


def get_parser():
    parser = argparse.ArgumentParser(
        description="Generate pretrain commands for linear-probe configs."
    )
    parser.add_argument(
        "-td",
        "--top_config_dir",
        default="configs/linear_probe",
        help="Config root (e.g. configs/linear_probe)",
    )
    parser.add_argument(
        "-o",
        "--output_path",
        default=str(GENERATED_DIR / "commands_linear_probe.sh"),
        help="Output shell script",
    )
    parser.add_argument(
        "-e",
        "--epochs",
        type=int,
        default=None,
        help="Pass -e to pretrain (caps rebar cross-attn stage too)",
    )
    parser.add_argument(
        "--seeds",
        default=",".join(str(s) for s in DEFAULT_SEEDS),
        help=f"Comma-separated seeds (default: {','.join(str(s) for s in DEFAULT_SEEDS)})",
    )
    parser.add_argument(
        "--save_root",
        default=None,
        help="If set, outputs go under save_root/{dataset}/{model}/seed_{N}",
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="Only one dataset subdir (har, ecg, ppg, eeg_2ch)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Include commands even when scores.pkl exists",
    )
    return parser


if __name__ == "__main__":
    args = get_parser().parse_args()
    seeds = tuple(int(s.strip()) for s in args.seeds.split(",") if s.strip())
    commands = build_linear_probe_commands(
        top_config_dir=args.top_config_dir,
        seeds=seeds,
        epochs=args.epochs,
        save_root=args.save_root,
        force=args.force,
        dataset=args.dataset,
    )
    output_path = write_command_script(commands, args.output_path)
    print(f"Wrote {len(commands)} commands to {output_path}")

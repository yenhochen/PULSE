"""Generate pretrain commands for limited-label supervised baselines."""

import argparse

from utils.experiment_commands import (
    DEFAULT_SEEDS,
    GENERATED_DIR,
    build_pretrain_commands,
    write_command_script,
)


def get_parser():
    parser = argparse.ArgumentParser(
        description="Generate pretrain commands for configs/semi-supervised."
    )
    parser.add_argument(
        "-td",
        "--top_config_dir",
        default="configs/semi-supervised",
        help="Config root (default: configs/semi-supervised)",
    )
    parser.add_argument(
        "-o",
        "--output_path",
        default=str(GENERATED_DIR / "commands_semi_supervised.sh"),
        help="Output shell script",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Include commands even when scores.pkl exists",
    )
    return parser


if __name__ == "__main__":
    args = get_parser().parse_args()
    commands = build_pretrain_commands(
        args.top_config_dir,
        seeds=DEFAULT_SEEDS,
        force=args.force,
    )
    output_path = write_command_script(commands, args.output_path)
    print(f"Wrote {len(commands)} commands to {output_path}")

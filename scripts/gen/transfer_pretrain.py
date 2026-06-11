"""Generate pretrain commands for transfer source-domain pretraining."""

import argparse

from utils.experiment_commands import (
    DEFAULT_SEEDS,
    GENERATED_DIR,
    build_pretrain_commands,
    write_command_script,
)


def get_parser():
    parser = argparse.ArgumentParser(
        description="Generate pretrain commands for transfer_pretrain configs."
    )
    parser.add_argument(
        "-td",
        "--top_config_dir",
        required=True,
        help="Config root (e.g. configs/transfer_pretrain/har)",
    )
    parser.add_argument(
        "-o",
        "--output_path",
        default=None,
        help="Output shell script (default: scripts/generated/commands_transfer_pretrain_<dataset>.sh)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Include commands even when checkpoint_best exists",
    )
    return parser


if __name__ == "__main__":
    args = get_parser().parse_args()
    dataset = args.top_config_dir.rstrip("/").split("/")[-1]
    output_path = args.output_path or GENERATED_DIR / f"commands_transfer_pretrain_{dataset}.sh"

    commands = build_pretrain_commands(
        args.top_config_dir,
        seeds=DEFAULT_SEEDS,
        transfer=True,
        force=args.force,
        skip_marker="checkpoint_best/model_state.pt",
    )
    output_path = write_command_script(commands, output_path)
    print(f"Wrote {len(commands)} commands to {output_path}")

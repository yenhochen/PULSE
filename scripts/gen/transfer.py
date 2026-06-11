"""Generate fine-tuning commands for transfer learning."""

import argparse

from utils.experiment_commands import (
    GENERATED_DIR,
    build_transfer_commands,
    write_command_script,
)


def get_parser():
    parser = argparse.ArgumentParser(
        description="Generate scripts/run/transfer.py commands from pretrained checkpoints."
    )
    parser.add_argument(
        "-td",
        "--top_backbone_dir",
        required=True,
        help="Pretrained run root (e.g. experiments/transfer/har)",
    )
    parser.add_argument(
        "-tc",
        "--transfer_config",
        required=True,
        help="Target yaml (e.g. configs/transfer/epilepsy.yaml)",
    )
    parser.add_argument(
        "-o",
        "--output_path",
        default=None,
        help="Output shell script",
    )
    return parser


if __name__ == "__main__":
    args = get_parser().parse_args()
    target = args.transfer_config.split("/")[-1].replace(".yaml", "")
    output_path = args.output_path or GENERATED_DIR / f"commands_transfer_{target}.sh"

    commands = build_transfer_commands(args.top_backbone_dir, args.transfer_config)
    output_path = write_command_script(commands, output_path)
    print(f"Wrote {len(commands)} commands to {output_path}")

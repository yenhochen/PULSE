# Smoke test

Short run of all linear-probe configs (train + full downstream eval). Rebar is skipped by default.

```bash
python test_scripts/run_linear_probe_all.py --epochs 2 --seeds 0 --run
```

Output: `experiments/linear_probe_check/`. Add `--include-rebar` to include Rebar. Other flags: `--dataset har`, `--force`, `--dry-run`.

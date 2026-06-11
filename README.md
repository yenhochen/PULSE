# PULSE

Authors: Yenho Chen, Maxwell A. Xu, James M. Rehg, Christopher J. Rozell

Code for PULSE (ICML 2026), a self-supervised pretraining method for physiological time-series based on preserving dynamical systems structure while filtering irrelevant noise. This is accomplished through a cross-reconstruction objective that estimates dynamics given a time series, then uses those dynamics to reconstruct a different randomly sampled time-series from the same system. In doing so, we separate transferable from non-transferable information in time-series data, improving label efficiency and transfer learning performance across a wide range of real-world datasets.

Paper: [arxiv.org/pdf/2512.00239](https://arxiv.org/pdf/2512.00239)

## Setup

Run commands from the repo root with `PYTHONPATH` set. Evaluation uses [cuML](https://docs.rapids.ai/) (RAPIDS 25.8 in `environment.yaml`).

```bash
git clone https://github.com/yenhochen/PULSE.git
cd PULSE

conda env create -f environment.yaml
conda activate pulse
export PYTHONPATH=$(pwd)
```


## Data

Scripts live in `data/process/`. Processed data is not bundled with the repo.

```bash
# HAR (other datasets: data/process/{ecg,ppg,sleepeeg,...}_processdata.py)
# Output: data/har/processed/
python -m data.process.har_processdata

# Synthetic analysis (optional): Lorenz, Thomas, Hindmarsh-Rose trajectories
# Output: data/analysis/{system}/noise-{level}/{params}/
sh data/process/analysis/build.sh
```

## Quick start (linear probe)

Pretrain PULSE on HAR and run downstream linear-probe eval (`configs/linear_probe/`):

```bash
python scripts/run/pretrain.py -c configs/linear_probe/har/pulse.yaml -s 0 \
  -sd experiments/har/pulse/seed_0
```

## Transfer learning

Transfer targets ([TFC-pretraining](https://github.com/mims-harvard/TFC-pretraining) on Figshare). Article zip URLs return empty files; use:

```bash
python -m data.process.download_transfer_data
# → data/epilepsy/processed/ and data/gesture/processed/
```

Sources: [Epilepsy](https://figshare.com/articles/19930199), [Gesture](https://figshare.com/articles/19930247).

```bash
# Pretrain on source domain (HAR)
# Output: experiments/transfer/har/pulse/seed_0/
python scripts/run/pretrain.py -c configs/transfer_pretrain/har/pulse.yaml -s 0 \
  -sd experiments/transfer/har/pulse/seed_0

# Fine-tune on target domain (Epilepsy)
python scripts/run/transfer.py -c configs/transfer/epilepsy.yaml \
  -p experiments/transfer/har/pulse/seed_0/checkpoint_best -s 0
```

## Layout

```
configs/         experiment YAML
data/process/    preprocessing
pulse/           PULSE modules
trainers/        model trainers
utils/           data, eval, I/O
scripts/run/     pretrain, transfer, semisupervised
scripts/gen/     batch command generators
test_scripts/    pipeline sanity check (see test_scripts/README.md)
```

Outputs are written to `experiments/{dataset}/{model}/seed_{N}/`.

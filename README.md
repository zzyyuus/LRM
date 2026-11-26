# LRMNet

PyTorch implementation of **LRMNet: Lightweight Intra-Image Region Relation Modeling for Remote Sensing Scene Classification**, accepted for publication in IEEE Signal Processing Letters.

LRMNet combines lightweight convolution with Mamba-based spatial relation modeling. A backbone of depthwise-separable BasicBlocks extracts hierarchical local representations, followed by parallel channel and spatial recalibration. Recalibrated features are compressed into spatial tokens, processed by a selective state-space block, and fused with convolutional features through a spatially gated residual connection.

![Architecture](assets/architecture.png)

## Project Structure

```text
LRM/
├── assets/                     # Architecture illustration
├── code/
│   ├── our_models.py            # Backbone and Mamba components
│   ├── bottleneck_attention.py  # Compressed contextual refinement
│   ├── experiment_models.py     # Model variants
│   ├── ablation_mixers.py       # Alternative contextual mixers
│   ├── targeted_mixup.py        # Training augmentation
│   ├── train_experiment.py      # Main training and evaluation runner
│   └── train_rs.py              # Dataset and training utilities
├── configs/                    # Original experiment configurations
├── splits/                     # Dataset audit and fixed partitions
├── requirements.txt
└── README.md
```

## Installation

```bash
git clone https://github.com/zzyyuus/LRM.git
cd LRM
pip install -r requirements.txt
```

## Dataset Preparation

Experiments use **SIRI-WHU** and **UC Merced Land Use (UCMerced)**. Prepare the images locally. Dataset partitions are provided in `splits/`.

Set `--data-root` to the directory relative to which the image paths in the selected split file resolve. For example, `UCMerced_bundle_fold1.json` contains paths such as `UCMerced_LandUse/Images/agricultural/agricultural63.tif`, so the data root should contain the `UCMerced_LandUse` folder. SIRI-WHU paths follow the directory structure recorded in its split file.

## Training

Run commands from the repository root. Replace `/path/to/data` with your data root. Original experiment configurations are retained in `configs/`; their server-specific paths must be adapted if reused.

```bash
python code/train_experiment.py --config configs/ucmerced.json --data-root /path/to/data --output runs/UCMerced_s42
python code/train_experiment.py --config configs/siri_whu.json --data-root /path/to/data --output runs/SIRI_s42
```

Training settings are stored in the configuration files. Command-line arguments override configuration values.

For the three-seed experiments, repeat with `--seed 43` and `--seed 44`, using a different output directory for each run. For the module-free baseline, add `--method l4_without_attention` and use a separate output directory.

## Testing and Evaluation

The same runner performs testing automatically after training. The commands above select the EMA checkpoint using validation performance and evaluate that checkpoint on the test split through `--test-selected-only`. No separate evaluation script is required.

Results are saved in the directory passed to `--output`:

| File | Content |
|---|---|
| `result.json` | Test OA, macro-F1, Kappa, confusion matrix, and run summary |
| `metrics.json` | Per-epoch training and validation metrics |
| `config.json` | Training configuration |
| `split.json` | Dataset partition used by the run |
| `best_ema.pt` | EMA checkpoint selected by validation accuracy |
| `last.pt` | Checkpoint for resuming training |

Accuracy and macro-F1 in the JSON results are fractions; multiply by 100 for percentages. An output directory containing `result.json` is treated as a completed run. Use a new directory for a new experiment.

To resume an interrupted run, repeat its original command with the same output directory and add `--resume-from /path/to/last.pt`.

## Citation

**LRMNet: Lightweight Intra-Image Region Relation Modeling for Remote Sensing Scene Classification.** IEEE Signal Processing Letters, accepted for publication.

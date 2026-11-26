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

- [Training script](code/train_experiment.py)
- [UCMerced configuration](configs/screen/UCMerced_f1_l4_resgate_spatial_k128_geo_s42/config.json)
- [SIRI-WHU configuration](configs/SIRI-WHU.json)
- [Dataset splits](splits/)

## Testing and Evaluation

Testing and evaluation are included in the [training script](code/train_experiment.py).

## Model Weights

| Dataset | Seed 42 | Seed 43 | Seed 44 |
|---|---|---|---|
| SIRI-WHU | [Download](weights/SIRI-WHU/seed42.pt) | [Download](weights/SIRI-WHU/seed43.pt) | [Download](weights/SIRI-WHU/seed44.pt) |
| UCMerced | [Download](weights/UCMerced/seed42.pt) | [Download](weights/UCMerced/seed43.pt) | [Download](weights/UCMerced/seed44.pt) |

## Citation

**LRMNet: Lightweight Intra-Image Region Relation Modeling for Remote Sensing Scene Classification.** IEEE Signal Processing Letters, accepted for publication.

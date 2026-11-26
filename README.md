# LRMNet

PyTorch implementation of **LRMNet: Lightweight Intra-Image Region Relation Modeling for Remote Sensing Scene Classification**, accepted for publication in **IEEE Signal Processing Letters**.

LRMNet combines a depthwise-separable residual backbone with parallel channel/spatial recalibration, compressed spatial-token processing, and outer spatially gated residual fusion. The paper configuration uses a ResNet-18 topology and a contextual feature width of 128.

![LRMNet architecture](assets/architecture.png)

This release includes model and ablation implementations, training and evaluation entry points, experiment configurations, and fixed dataset splits. Trained checkpoints, raw training logs, and dataset images are not distributed.

## Installation

```bash
git clone https://github.com/zzyyuus/LRM.git
cd LRM
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Install a matching PyTorch/torchvision build for your device using the [PyTorch installation guide](https://pytorch.org/get-started/locally/). The included selective state-space implementation uses PyTorch operations; it does not import the external `mamba-ssm` package. Runtime validation of this cleaned release is pending. The dependency file specifies minimum versions, not a locked environment.

## Dataset preparation

Obtain SIRI-WHU and UC Merced Land Use from their dataset providers. Images are not included. Pass the directory immediately containing the class folders as `--data-root`:

```text
dataset_root/
  agricultural/    # UC Merced example
    agricultural00.tif
    ...
  airplane/
    ...
```

SIRI-WHU uses its own class folders, such as `agriculture`. Do not rename the classes or images. `splits/*_fold1.json` specifies the exact filenames, class indices, and train/validation/test membership. Paths have been reduced to `class/filename` for portability; split membership and labels are unchanged. The launcher checks that all referenced images exist before starting training.

## Training

Run commands from the repository root:

```bash
python train.py --config configs/UCMerced.json --data-root /path/to/UCMerced_LandUse/Images --output runs/UCMerced_s42 --seed 42 --device cuda
python train.py --config configs/SIRI-WHU.json --data-root /path/to/12class_tif --output runs/SIRI_s42 --seed 42 --device cuda
```

Repeat with seeds **43** and **44**, using a different output directory for each run. `--device cpu`, `mps`, and `auto` are also supported. CPU training may be slow. Configurations retain dataset-specific settings from the source experiment records:

| Setting | SIRI-WHU | UCMerced |
|---|---|---|
| Backbone depth / context width | 18 / 128 | 18 / 128 |
| Batch size | 16 | 64 |
| Augmentation profile | paper | geo |
| Additional mixing | hw_mixup | none |
| Learning rate / optimizer | 0.0001 / Adam | 0.0001 / Adam |
| EMA decay | 0.99 | 0.99 |
| Maximum epochs / patience | 300 / 30 | 300 / 30 |
| Selected checkpoint | best_ema.pt | best_ema.pt |

The original runner is retained in `code/train_experiment.py`. `train.py` loads the portable configuration and fixed split without requiring an archived server audit file. A run writes its configuration, metrics, selected checkpoint, and result to the specified output directory. These generated artifacts are ignored by Git. Existing experiment results cause the runner to skip an already completed output directory.

## Evaluation

Checkpoints must be produced locally using this implementation:

```bash
python evaluate.py --config configs/UCMerced.json --checkpoint runs/UCMerced_s42/best_ema.pt --data-root /path/to/UCMerced_LandUse/Images --device cuda
```

For SIRI-WHU, use `configs/SIRI-WHU.json` and the corresponding data root/checkpoint. Evaluation prints loss, overall accuracy, macro-F1, Kappa, and the confusion matrix. Accuracy and macro-F1 are fractions; multiply by 100 to express percentages. Checkpoint selection and early stopping use validation data. The supplied configurations request test evaluation of the selected checkpoint only.

## Paper-reported results

| Dataset | OA (%) | Parameters |
|---|---|---|
| SIRI-WHU | 93.23 ± 0.51 | 1,930,637 |
| UCMerced | 89.60 ± 1.07 | 1,939,862 |

OA values are paper-reported three-seed results, not results from rerunning this release. The reported 0.724 counted GFLOPs covers convolution and matrix multiplication, with one MAC counted as two FLOPs; it excludes state-space/custom scans and is not a complete operation count. Hardware and library differences may affect reproducibility.

## Code organization

```text
train.py                    portable training launcher
evaluate.py                 checkpoint evaluation
code/our_models.py          backbone and sequence-processing components
code/bottleneck_attention.py compressed contextual refinement
code/experiment_models.py   model factory and variants
code/ablation_mixers.py     alternative contextual mixers
configs/                    dataset-specific training settings
splits/                     fixed fold-1 partitions
assets/                     architecture illustration
```

The paper model factory call is `build_model("l4_resgate_spatial_k128", 18, num_classes, 128)`. Other factories in the archived code are not the default paper model. Historical internal class names are retained to preserve compatibility. Ablation implementations are included; the two provided configurations target the complete model.

## Citation

Please cite **LRMNet: Lightweight Intra-Image Region Relation Modeling for Remote Sensing Scene Classification**, IEEE Signal Processing Letters, accepted for publication. Final DOI and bibliographic metadata will be added when available.

## Acknowledgements

Related research: [Mamba](https://github.com/state-spaces/mamba), [Vision Mamba](https://github.com/hustvl/Vim), [VMamba](https://github.com/MzeroMiko/VMamba), and [RSMamba](https://github.com/KyanChen/RSMamba).



The earlier single-file implementation in lrmnet/LRMNet.py is retained for history. Use the root train.py and evaluate.py entry points with configs/ for the current release.

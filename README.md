# LRMNet

PyTorch implementation of **LRMNet: Lightweight Intra-Image Region Relation Modeling for Remote Sensing Scene Classification**, accepted for publication in IEEE Signal Processing Letters.

LRMNet combines lightweight convolution with Mamba-based spatial relation modeling. A backbone of depthwise-separable BasicBlocks extracts hierarchical local representations, followed by parallel channel and spatial recalibration. Recalibrated features are compressed into spatial tokens, processed by a selective state-space block, and fused with convolutional features through a spatially gated residual connection.

![Architecture](assets/architecture.png)

## 安装

```bash
git clone https://github.com/zzyyuus/LRM.git
cd LRM
pip install -r requirements.txt
```

## 数据集

SIRI-WHU 和 UCMerced。请自行准备图像数据，数据划分位于 `splits/`。

## 使用

模型和训练源码位于 `code/`，训练配置位于 `configs/`。配置中的原始绝对路径需根据本地目录调整。

以下为 UCMerced 主模型种子 42 的训练命令。将 `/path/to/data` 换成包含划分文件中相对图像路径的数据根目录。其他两种子使用 43 和 44。去模块基线使用 `--method l4_without_attention`。

```bash
python code/train_experiment.py \
  --audit splits/audit.json --dataset UCMerced \
  --split-file splits/UCMerced_bundle_fold1.json --data-root /path/to/data \
  --output runs/UCMerced_lrmnet_s42 --method l4_resgate_spatial_k128 \
  --depth 18 --width 128 --seed 42 --split-seed 20260929 --device cuda \
  --epochs 300 --min-epochs 50 --patience 30 --batch-size 64 --workers 0 \
  --augmentation none --aug-profile geo --ema-decay 0.99 --lr 0.0001 \
  --select ema_best --early-stop-metric ema_val --test-selected-only
```

训练命令中的 `--test-selected-only` 对验证集选定的权重进行测试，结果包含 OA、macro-F1 和 Kappa。

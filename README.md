<div align="center">

# Vision Transformer

Replicating Dosovitskiy et al. (ICLR 2021) on Tiny ImageNet

*PyTorch · From Scratch & Pretrained*

[arxiv.org/abs/2010.11929](https://arxiv.org/abs/2010.11929)

</div>

---

## Abstract

This project implements the Vision Transformer (ViT-B/16) from first principles, following the original paper equations directly. Two strategies are benchmarked on Tiny ImageNet: training from scratch and fine-tuning a pretrained backbone. Results confirm the paper's central finding — ViT is data-hungry by design, and large-scale pretraining is decisive at small dataset scales.

---

## Results

```
ViT-B/16 — trained from scratch                  9.47%
ViT-B/16 — pretrained (ImageNet-21k), fine-tuned  84.06%
```

The 74-point gap reflects ViT's lack of convolutional inductive bias. With only 100K training images, the model cannot learn translation equivariance from scratch — pretraining on 14M images closes that gap entirely.

---

## Dataset

Tiny ImageNet — 200-class subset of ImageNet.
100,000 training images · 10,000 validation images · 64×64 RGB resolution.

---

## Approach

### Patch Embedding

Images are divided into non-overlapping patches using `Conv2d(kernel=stride=P)`, mathematically equivalent to the linear projection E in Eq. 1. For 64×64 input with `patch_size=8`, this produces N = 64 patch tokens each projected to D = 768.

### Transformer Encoder

12 layers, each with pre-norm Multi-Head Self-Attention (12 heads) followed by a pre-norm MLP block (hidden dim 3072, GELU activation), both with residual connections — implementing Eq. 2 and Eq. 3.

### Classification

The CLS token at the final layer is passed through LayerNorm → Linear(768, 200) — Eq. 4.

### Attention Rollout

Post-training saliency is computed via attention rollout (Abnar & Zuidema, 2020). At each layer, `Â = 0.5·A + 0.5·I` (accounting for the residual), rolled across all 12 layers to produce a single CLS-to-patch attention map.

---

## Training

From scratch

```
Image size      64×64
Patch size      8  (N=64 tokens)
Optimizer       AdamW
LR              1e-3
Weight decay    0.05
Label smooth    0.1
Schedule        Cosine + 4-epoch linear warmup
Batch size      32
Epochs          10
```

Feature extraction

```
Weights         ViT_B_16_Weights.DEFAULT (ImageNet-21k)
Backbone        frozen
Head            nn.Linear(768, 200)
Optimizer       Adam
LR              1e-3
Schedule        Cosine + 2-epoch linear warmup
Epochs          10
```

---

## Structure

```
vit.ipynb              —  implementation, training, attention visualisation
dataset.py             —  TinyImageNet Dataset class
data/tiny-imagenet-200 —  train / val split (not tracked)
```

---

## Setup

```bash
pip install torch torchvision numpy matplotlib Pillow tqdm ipykernel jupyter
```

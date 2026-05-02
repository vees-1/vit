<div align="center">

# Vision Transformer From Scratch

**ViT-B/16 implementation and Tiny ImageNet experiment in PyTorch**

[![Repository](https://img.shields.io/badge/GitHub-vit-black?style=flat-square)](https://github.com/vees-1/vit)
[![Paper](https://img.shields.io/badge/Paper-An%20Image%20is%20Worth%2016x16%20Words-blue?style=flat-square)](https://arxiv.org/abs/2010.11929)

</div>

---

## What Is This Project?

This project implements a Vision Transformer from scratch in PyTorch and tests it on Tiny ImageNet.

The goal is to understand the original ViT paper by building the important pieces manually:

- patch embedding,
- class token,
- positional embedding,
- transformer encoder blocks,
- MLP blocks,
- final classification head,
- training loop,
- evaluation,
- attention rollout visualization.

The project also compares two approaches:

- training ViT-B/16 from scratch,
- using a pretrained ViT-B/16 backbone and fine-tuning only the classifier head.

---

## Why This Project Exists

CNNs have built-in image assumptions such as locality and translation equivariance. Vision Transformers do not have the same convolutional bias. They treat images as sequences of patches and learn relationships through self-attention.

This project shows that idea clearly:

- a ViT can be built using standard transformer components,
- patch embeddings turn an image into tokens,
- the class token becomes the final image representation,
- training from scratch on a small dataset is difficult,
- large-scale pretraining makes ViT much more effective.

This is the same core lesson from the original ViT paper.

---

## Dataset

The experiment uses Tiny ImageNet.

Tiny ImageNet contains:

- 200 classes,
- 100,000 training images,
- 10,000 validation images,
- 64 x 64 RGB images.

Local dataset path:

```text
data/tiny-imagenet-200/
```

The dataset is not committed to the repository.

---

## Results

| Strategy | Best Validation Accuracy |
| --- | ---: |
| ViT-B/16 trained from scratch for 10 epochs | `9.47%` |
| Pretrained ViT-B/16 feature extractor fine-tuned for 10 epochs | `84.06%` |

The large gap is the main learning outcome of the project.

Training from scratch performs poorly because Tiny ImageNet is too small for a large ViT to learn strong image representations from random initialization. The pretrained model performs much better because it already learned useful visual features from large-scale data.

---

## Architecture

```text
Tiny ImageNet image
   |
   v
Resize and normalize
   |
   v
Patch embedding
   |
   |-- Conv2d with kernel size = patch size
   |-- stride = patch size
   |-- produces patch tokens
   v
Token sequence
   |
   |-- prepend learnable class token
   |-- add learnable positional embedding
   v
Transformer encoder
   |
   |-- LayerNorm
   |-- multi-head self-attention
   |-- residual connection
   |-- LayerNorm
   |-- MLP block
   |-- residual connection
   v
CLS token
   |
   v
LayerNorm + Linear classifier
   |
   v
200-class Tiny ImageNet prediction
```

---

## What Has Been Built

### Dataset Loader

Implemented in `dataset.py`.

The dataset loader:

- reads Tiny ImageNet class IDs from `wnids.txt`,
- maps WordNet IDs to class indexes,
- reads human-readable class names from `words.txt`,
- loads training images from class folders,
- loads validation labels from `val_annotations.txt`,
- returns RGB image and class label pairs.

### ViT From Scratch

Implemented in `vit.ipynb`.

The notebook builds:

- `PatchEmbedding`,
- `MultiheadSelfAttentionBlock`,
- `MLPBlock`,
- `TransformerEncoderBlock`,
- `ViT`,
- training loop,
- validation loop,
- cosine learning-rate schedule with warmup.

The scratch model uses:

```text
Image size: 64 x 64
Patch size: 8
Patch tokens: 64
Embedding dim: 768
Transformer layers: 12
Attention heads: 12
MLP hidden dim: 3072
Classes: 200
Parameters: 85,408,712
```

### Pretrained ViT Experiment

The notebook also uses:

```text
torchvision.models.ViT_B_16_Weights.DEFAULT
```

The pretrained backbone is frozen, and only the classifier head is trained for Tiny ImageNet.

This setup has:

```text
Total parameters: 85,952,456
Trainable parameters: 153,800
Trainable share: 0.2%
```

### Attention Rollout

The notebook includes attention rollout to visualize where the transformer attends.

The method combines attention across layers and accounts for residual connections:

```text
A_hat = 0.5 * attention + 0.5 * identity
```

This creates a class-token-to-patch attention map for visual interpretation.

---

## Repository Map

```text
vit.ipynb
  Main notebook with ViT implementation, training, pretrained comparison,
  evaluation, and attention visualization

dataset.py
  Tiny ImageNet dataset class

data/tiny-imagenet-200/
  Local dataset folder
  Not intended to be committed
```

---

## Training Setup

### From Scratch

```text
Optimizer: AdamW
Learning rate: 1e-3
Weight decay: 0.05
Label smoothing: 0.1
Schedule: cosine with 4-epoch warmup
Batch size: 32
Epochs: 10
Device used in notebook: MPS
```

### Pretrained Feature Extractor

```text
Backbone: ViT-B/16 pretrained weights
Backbone training: frozen
Classifier head: Linear(768, 200)
Optimizer: Adam
Learning rate: 1e-3
Schedule: cosine with 2-epoch warmup
Epochs: 10
```

---

## Setup

Install dependencies:

```bash
pip install torch torchvision numpy matplotlib Pillow tqdm ipykernel jupyter
```

Run the notebook:

```bash
jupyter notebook vit.ipynb
```

Expected local dataset path:

```text
data/tiny-imagenet-200/
```

---

## Current Limitations

- The project is notebook-first.
- The training code is not yet packaged into reusable modules.
- The dataset must be downloaded manually.
- No experiment tracking tool is included.
- The scratch model is large for Tiny ImageNet and only trained for 10 epochs.
- No hyperparameter sweep is included.

---

## Future Plans

- Move model classes into a separate Python module.
- Add a training CLI with configurable hyperparameters.
- Add checkpoint saving and loading.
- Add experiment logging.
- Try smaller ViT variants better suited to Tiny ImageNet.
- Train longer from scratch with stronger regularization.
- Compare against a simple CNN baseline.
- Add more attention visualization examples.
- Add reproducible seed/config files.

---

## Project Status

This project demonstrates the full ViT learning pipeline:

```text
image
  -> patch tokens
  -> transformer encoder
  -> class token
  -> classifier
  -> Tiny ImageNet prediction
```

It is useful as a resume project because it shows that the model was understood and implemented from first principles, not only imported from a library.

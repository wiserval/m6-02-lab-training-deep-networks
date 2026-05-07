![logo_ironhack_blue 7](https://user-images.githubusercontent.com/23629340/40541063-a07a0a8a-601a-11e8-91b5-2f13e4e6b441.png)

# Lab | Training Deep Networks

## Overview

You now know what a network *is*. Today you'll learn to make one *learn*. The goal of this lab is to wire up the canonical PyTorch training loop, watch it succeed, watch it fail, and use the regularisation toolkit from the lesson to fix the failures.

You'll train MLPs on **Fashion-MNIST** — a drop-in replacement for the classic MNIST digit dataset, but with 28×28 grayscale images of clothing items in 10 classes. It's small enough to train on a laptop and rich enough to make overfitting and learning-rate problems visible.

## Learning Goals

By the end of this lab you should be able to:

- Implement a complete PyTorch training loop with mini-batches, loss tracking, and validation.
- Diagnose underfitting vs overfitting by reading training and validation loss curves.
- Apply dropout, batch normalisation, and early stopping to a real model and quantify their effect.
- Compare optimisers (SGD vs Adam) and learning-rate schedules on the same task.

## Setup and Context

You'll work in a single Jupyter Notebook. Fashion-MNIST is available through `torchvision.datasets.FashionMNIST` and downloads automatically.

## Requirements

### Fork and clone

1. Fork this repository to your own GitHub account.
2. Clone the fork to your local machine.
3. Navigate into the project directory.

### Python environment

```bash
pip install numpy pandas matplotlib torch torchvision
```

If you have a GPU available, PyTorch will use it automatically when you call `.to(device)`. CPU works fine for this lab — expect a couple of minutes per training run.

## Getting Started

1. Create a notebook called **`m6-02-training-deep-networks.ipynb`**.
2. Standard imports:

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(42)
np.random.seed(42)
```

3. Load Fashion-MNIST and create train and validation `DataLoader`s with batch size 128.

```python
tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_set = datasets.FashionMNIST(root="./data", train=True, download=True, transform=tf)
test_set = datasets.FashionMNIST(root="./data", train=False, download=True, transform=tf)
train_loader = DataLoader(train_set, batch_size=128, shuffle=True)
val_loader = DataLoader(test_set, batch_size=256, shuffle=False)
```

## Tasks

### Task 1 — Build the Training Loop

1. Define a simple MLP for Fashion-MNIST:
   - Input: flatten the 28×28 image into 784 features.
   - Hidden: 256 → ReLU → 128 → ReLU.
   - Output: 10 logits (no softmax — `CrossEntropyLoss` includes it).
2. Implement the canonical training loop. For each epoch:
   - Iterate over `train_loader` with the five-step loop (`zero_grad`, forward, loss, backward, step).
   - At the end of the epoch compute the **average training loss** and the **validation loss + accuracy** on `val_loader`.
3. Train for 15 epochs with `Adam(lr=1e-3)`. Save the per-epoch train and validation losses.
4. Plot training and validation loss on the same axes.

**Expected behaviour:** validation accuracy should reach roughly 87–89% in this baseline setup.

### Task 2 — Make It Overfit On Purpose

Now break the model deliberately so you can see what overfitting looks like.

1. Reduce the training set to **1000 examples** by taking the first 1000 rows of `train_set`.
2. Build a much larger MLP: hidden sizes `[512, 512, 512]`, all ReLU.
3. Train for 50 epochs with the same `Adam(lr=1e-3)`.
4. Plot training and validation loss. The training loss should go to near zero while validation loss climbs back up.
5. Report the gap between final training and validation accuracy.

In a markdown cell, write 2–3 sentences interpreting the curves: where does the model start overfitting, and how can you tell?

### Task 3 — Fight Overfitting

Take the same overfitting setup from Task 2 (1000-sample training set, large MLP) and apply the regularisation toolkit. For each technique below, train a new model from scratch and report the best validation accuracy and the train/val gap.

1. **Dropout** with `p=0.3` after each hidden layer.
2. **Batch normalisation** with `nn.BatchNorm1d` after each linear layer (before the activation).
3. **Weight decay** in the optimiser: `Adam(..., weight_decay=1e-3)`.
4. **Early stopping** — keep tracking the best validation loss; if it doesn't improve for 5 epochs, stop and roll back to the best weights.

Tabulate the results:

| Technique | Best val accuracy | Train/val gap | Final epoch |
|---|---|---|---|
| Baseline (Task 2) | … | … | … |
| Dropout | … | … | … |
| BatchNorm | … | … | … |
| Weight decay | … | … | … |
| Early stopping | … | … | … |

In a markdown cell, answer: which technique gave the largest improvement on this small dataset, and what does that suggest?

### Task 4 — Optimisers and Learning Rate

Go back to the **full** Fashion-MNIST training set and the original two-hidden-layer MLP from Task 1.

1. Train three identical models for 15 epochs each, varying only the optimiser:
   - `SGD(lr=0.01, momentum=0.9)`
   - `Adam(lr=1e-3)`
   - `Adam(lr=1e-4)`
2. Plot the validation loss curves for all three on the same axes.
3. In a markdown cell, comment on:
   - Which optimiser converges fastest in the first few epochs?
   - Which one ends up with the lowest validation loss?
   - What does the comparison between the two Adam runs tell you about learning-rate sensitivity?

### Task 5 — Add a Learning-Rate Schedule (Stretch)

This task is optional but recommended.

1. Take the best model from Task 4 and add a `torch.optim.lr_scheduler.CosineAnnealingLR` scheduler with `T_max=15`.
2. Train for 15 epochs and plot the learning rate over time alongside the loss curves.
3. Report whether cosine annealing improved the final validation accuracy compared to the same model with a constant learning rate.

## Submission

### What to submit

- `m6-02-training-deep-networks.ipynb` — completed notebook.

### Definition of done (checklist)

- [ ] Working training loop with both training and validation loss tracking per epoch.
- [ ] Overfitting demonstration with clearly diverging train/val loss curves.
- [ ] Comparison table of regularisation techniques with at least 4 rows.
- [ ] Optimiser comparison with all three runs plotted on the same axes.
- [ ] Each task has at least one markdown cell with interpretation.
- [ ] `Kernel → Restart & Run All` produces no errors.

### How to submit (Git workflow)

```bash
git add .
git commit -m "lab: complete training deep networks"
git push origin main
```

Then open a **Pull Request** on the original repository describing what worked, what surprised you, and what you'd try next.

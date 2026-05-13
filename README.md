![logo_ironhack_blue 7](https://user-images.githubusercontent.com/23629340/40541063-a07a0a8a-601a-11e8-91b5-2f13e4e6b441.png)

# Lab | Training Deep Networks

## Overview

You now know what a neural network *is*. Today you'll make one *learn*. In this lab you'll train **one** fully-specified neural network on **Fashion-MNIST** — a drop-in replacement for the classic MNIST digit dataset, but with 28×28 grayscale images of clothing items in 10 classes. Every piece of the recipe is given to you: the exact architecture, the loss, the optimiser, the learning-rate schedule, the regularisation, and the training-time hyperparameters. Your job is to wire up the canonical PyTorch training loop, run it, and read the curves.

The point is not to compare ten variants of the same model. The point is to get **one** clean, modern training run end to end — every component from the lesson present in a single, consistent recipe.

## Learning Goals

By the end of this lab you should be able to:

- Translate a precisely specified network architecture into PyTorch (`nn.Sequential` or a custom `nn.Module`).
- Wire up a canonical PyTorch training loop with mini-batches, validation, an optimiser, and a learning-rate scheduler.
- Track and plot training and validation loss and accuracy per epoch.
- Read those curves and write a brief, honest interpretation of what the training run produced.

## Setup and Context

You'll work in a single Jupyter Notebook. Fashion-MNIST is available through `torchvision.datasets.FashionMNIST` and downloads automatically. CPU is fine — the whole training run takes about 2–3 minutes on a modern laptop.

## Requirements

### Fork and clone

1. Fork this repository to your own GitHub account.
2. Clone the fork to your local machine.
3. Navigate into the project directory.

### Python environment

```bash
pip install numpy matplotlib torch torchvision
```

## Getting Started

1. Create a notebook called **`m6-02-training-deep-networks.ipynb`**.
2. Standard imports and seed:

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

3. Load Fashion-MNIST and create train/validation `DataLoader`s:

```python
tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_set = datasets.FashionMNIST(root="./data", train=True,  download=True, transform=tf)
val_set   = datasets.FashionMNIST(root="./data", train=False, download=True, transform=tf)
train_loader = DataLoader(train_set, batch_size=128, shuffle=True)
val_loader   = DataLoader(val_set,   batch_size=256, shuffle=False)
```

## Tasks

### Task 1 — Train the Network

You will train **one** classifier on Fashion-MNIST. Everything you need is fully specified below.

**The architecture.** A four-layer fully-connected ReLU network with batch normalisation on the hidden layers and dropout on the two widest layers. Input is a flattened 28×28 grayscale image (784 features); output is 10 class logits (no softmax — `CrossEntropyLoss` includes it).

| Step | Layer | Output shape | Notes |
|---|---|---|---|
| 0 | Input | `(B, 1, 28, 28)` | a Fashion-MNIST batch |
| 1 | `Flatten()` | `(B, 784)` | 28·28 = 784 features |
| 2 | `Linear(784, 256)` → `BatchNorm1d(256)` → `ReLU` → `Dropout(p=0.3)` | `(B, 256)` | widest hidden layer |
| 3 | `Linear(256, 128)` → `BatchNorm1d(128)` → `ReLU` → `Dropout(p=0.3)` | `(B, 128)` | second hidden layer |
| 4 | `Linear(128, 64)`  → `BatchNorm1d(64)`  → `ReLU` | `(B, 64)` | third hidden layer (no dropout here — kept narrow on purpose) |
| 5 | `Linear(64, 10)` | `(B, 10)` | class logits |

**The training-time hyperparameters.** Use exactly these values — no tuning, no comparisons, no extra variants.

| Parameter | Value |
|---|---|
| Loss | `nn.CrossEntropyLoss()` |
| Optimiser | `torch.optim.Adam` with `lr=1e-3`, `weight_decay=1e-4` |
| LR schedule | `torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)` |
| Epochs | 15 |
| Batch size | 128 (already set in the DataLoaders above) |
| Random seed | 42 (already set in the setup) |
| Device | `"cuda"` if available, otherwise `"cpu"` |

**What to do.**

1. Translate the architecture in the table into a PyTorch `nn.Sequential` (or a small `nn.Module` subclass — whichever you prefer) and move it to `device`. Print the model so the layers are visible in the notebook.
2. Build the canonical training loop. For each epoch:
   - Set the model to `train()` mode, iterate over `train_loader`, and run the five-step inner loop (`optimizer.zero_grad()`, forward, loss, `loss.backward()`, `optimizer.step()`). Call `scheduler.step()` once at the end of the epoch.
   - Set the model to `eval()` mode and, inside `torch.no_grad()`, compute the **average loss** and **accuracy** on both `train_loader` and `val_loader`.
   - Append `train_loss`, `val_loss`, `train_acc`, `val_acc` to four Python lists.
3. Run the loop for **15 epochs**. After training, plot two figures side by side: training and validation **loss** vs epoch, and training and validation **accuracy** vs epoch. Print the **best validation accuracy** and the **epoch** at which it occurred.
4. In a short markdown cell (2–3 sentences), describe what your curves look like — for example, does the model still improve at epoch 15, do training and validation loss stay close together, and what is your best validation accuracy. Validation accuracy is expected to land around **89–90%** on this architecture; report whatever number you actually got.

## Submission

### What to submit

- `m6-02-training-deep-networks.ipynb` — completed notebook.

### Definition of done (checklist)

- [ ] Network defined exactly as specified in the architecture table.
- [ ] Training loop runs for 15 epochs without errors and uses the cosine LR schedule.
- [ ] Per-epoch train/validation loss and accuracy stored and plotted.
- [ ] Best validation accuracy and the epoch it occurred at are printed.
- [ ] Short markdown cell interprets the curves.
- [ ] `Kernel → Restart & Run All` produces no errors.

### How to submit (Git workflow)

```bash
git add .
git commit -m "lab: complete training deep networks"
git push origin main
```

Then open a **Pull Request** on the original repository describing your final validation accuracy and what you noticed in the training curves.

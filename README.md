# Flower Classification CNN — PyTorch

A from-scratch learning project for understanding and building a Convolutional Neural Network with PyTorch on the [`flower_photos`](https://www.tensorflow.org/datasets/catalog/tf_flowers) dataset (5 flower classes, real-world RGB photos).

## Goal

Extend the CNN pipeline from grayscale digits to real-world color images, and practice the full workflow of preparing, training, and evaluating on a dataset that isn't handed to you pre-packaged:

```
Raw folder of images → Split into train/val/test → Augment → Convolution → ReLU → Pooling (×3) → Flatten → Dense → Classification
```

This project is also used to inspect what happens to the data before training (folder structure, class balance, sample images) and after training (misclassified examples, confusion matrix).

## Files

| File | Description |
|---|---|
| `split_dataset.py` | Splits the raw `data/flower_photos/<class>/` folders into `data/train/`, `data/val/`, `data/test/` (80/10/10) with a fixed random seed |
| `inspect_data.py` | Plots one sample image per class from the raw dataset, as a sanity check before splitting/training |
| `data.py` | Builds `ImageFolder` datasets from the split folders and wraps them in `DataLoader`s, with augmentation on the training set |
| `model.py` | The `FlowerCNN` architecture |
| `train.py` | Training loop (10 epochs, Adam) with per-epoch validation, saves weights to `flower_cnn.pth` |
| `evaluate.py` | Loads `flower_cnn.pth`, evaluates on the test set, and plots a confusion matrix |
| `visualize.py` | *(currently empty — placeholder for visualizing predictions/filters)* |
| `flower_cnn.pth` | Saved weights from a completed training run |
| `requirements.txt` | Dependencies |
| `.gitignore` | Excludes `data/` and `.pth` weight files from version control |

## The Data

The source dataset is the classic **`flower_photos`** set: 5 classes — `daisy`, `dandelion`, `roses`, `sunflowers`, `tulips` — as JPEGs of varying sizes, expected at `data/flower_photos/<class_name>/`.

`split_dataset.py` shuffles each class independently (seed `42`) and copies files into:

```
data/train/<class>/   (80%)
data/val/<class>/     (10%)
data/test/<class>/    (10%)
```

`data.py` then loads these with `torchvision.datasets.ImageFolder` and applies:

| Split | Transform |
|---|---|
| Train | `RandomResizedCrop(128×128, scale=0.8–1.0)` → `RandomHorizontalFlip()` → tensor → normalize (ImageNet mean/std) |
| Val / Test | `Resize(128×128)` → tensor → normalize (ImageNet mean/std) |

Random-resized cropping and horizontal flipping are applied **only** to the training split — this is data augmentation to reduce overfitting, and it would corrupt evaluation if applied to val/test. `DataLoader`s use `batch_size=32`, shuffled for training only.

## The Model

```python
nn.Sequential(
    nn.Conv2d(in_channels=3,  out_channels=16, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2),

    nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2),

    nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2),

    nn.Flatten(),
    nn.Linear(in_features=64*16*16, out_features=5)
)
```

**Shape trace** (128×128 RGB input, batch size 32):

| Layer | Output Shape |
|---|---|
| Input | `(32, 3, 128, 128)` |
| Conv2d (3→16) | `(32, 16, 128, 128)` |
| MaxPool2d | `(32, 16, 64, 64)` |
| Conv2d (16→32) | `(32, 32, 64, 64)` |
| MaxPool2d | `(32, 32, 32, 32)` |
| Conv2d (32→64) | `(32, 64, 32, 32)` |
| MaxPool2d | `(32, 64, 16, 16)` |
| Flatten | `(32, 16384)` |
| Linear (16384→5) | `(32, 5)` |

As in the MNIST version, `padding=1` on every `3×3` convolution keeps spatial size unchanged, so all downsampling comes from the three `MaxPool2d` layers: `128 → 64 → 32 → 16`. That's why `Linear` needs exactly `64*16*16 = 16384` input features — one three-stage pooling deeper than the MNIST model, since the images start 4.5× larger per side.

**Total trainable parameters: 105,509**

| Layer | Parameters |
|---|---|
| Conv2d (3→16) | 448 |
| Conv2d (16→32) | 4,640 |
| Conv2d (32→64) | 18,496 |
| Linear (16384→5) | 81,925 |

## Training Setup

| Setting | Value |
|---|---|
| Loss function | `CrossEntropyLoss` |
| Optimizer | `Adam`, `lr=0.001` |
| Epochs | 10 |
| Batch size | 32 |

The training loop follows the standard pattern: forward pass → compute loss → `zero_grad()` → `backward()` → `optimizer.step()`, accumulating average loss per epoch. Unlike the MNIST version, **validation runs at the end of every epoch** (not just once at the end) — loss and accuracy on `val_loader`, under `torch.no_grad()` — so overfitting is visible as training progresses rather than only after the fact. Final weights are saved to `flower_cnn.pth`.

## Evaluation

`evaluate.py` loads the saved weights, runs the model over the full test set, and reports:
- Overall test accuracy and raw correct/total count
- A **confusion matrix** (via `sklearn.metrics`) showing which flower classes get confused with each other

## Requirements

```
numpy
torch
torchvision
matplotlib
scikit-learn
pillow
```

```bash
pip install -r requirements.txt
```

## How to Run

```bash
# 1. Place the raw dataset at data/flower_photos/<class_name>/*.jpg
# 2. (optional) Sanity-check the raw folders
python inspect_data.py

# 3. Split into train/val/test
python split_dataset.py

# 4. Train (saves flower_cnn.pth on completion)
python train.py

# 5. Evaluate on the held-out test set
python evaluate.py
```

Expect training output in this shape:

```
Epoch 1/10 | Train Loss: ... | Val Loss: ... | Val Acc: ...
Epoch 2/10 | Train Loss: ... | Val Loss: ... | Val Acc: ...
...
```

And evaluation output like:

```
Test Accuracy: 0.XXXX
Correct: XXX/XXX
```
followed by a confusion matrix plot.

## What This Project Demonstrates

- Going from a raw, unsplit folder of real photos to a properly separated train/val/test pipeline — not just consuming an already-packaged dataset
- Why data augmentation (`RandomResizedCrop`, `RandomHorizontalFlip`) belongs on the training set only, never on validation/test
- Scaling the same conv → ReLU → pool block from grayscale 28×28 digits to color 128×128 photos, and how the `Linear` layer's input size still falls out directly from `(final channels) × (final height) × (final width)`
- Monitoring train vs. validation loss/accuracy every epoch to watch for overfitting as it happens, rather than only checking a single number at the end
- Using a confusion matrix, not just accuracy, to see *which* classes the model actually confuses (e.g. visually similar flowers)

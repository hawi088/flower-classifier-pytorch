"""
visualize.py — beautiful, animated visualizations for the Flower CNN project.

Produces four animated visualizations. Each one plays on screen and is also
saved as a GIF under outputs/:

    1. Training loss & accuracy curves   -> outputs/training_curves.gif
       (needs history.json, written by the updated train.py)
    2. Sample predictions grid           -> outputs/predictions_grid.gif
    3. Confusion matrix                  -> outputs/confusion_matrix.gif
    4. Conv1 filters + feature maps      -> outputs/filters_and_features.gif

Usage:
    python visualize.py                          # run all four
    python visualize.py --only curves predictions  # run a subset
"""

import argparse
import json
from pathlib import Path

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import confusion_matrix

from data import test_loader
from model import FlowerCNN

# ----------------------------------------------------------------------
# Style
# ----------------------------------------------------------------------

BG = "#12141c"
PANEL = "#1b1e2b"
GRID = "#2a2e40"
TEXT = "#e8e9f0"
TEAL = "#2ee6c5"
CORAL = "#ff6b6b"
GOLD = "#ffce54"

plt.rcParams.update({
    "figure.facecolor": BG,
    "axes.facecolor": PANEL,
    "axes.edgecolor": GRID,
    "axes.labelcolor": TEXT,
    "text.color": TEXT,
    "xtick.color": TEXT,
    "ytick.color": TEXT,
    "grid.color": GRID,
    "font.size": 11,
})

OUT_DIR = Path("outputs")
OUT_DIR.mkdir(exist_ok=True)

MEAN = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])


def unnormalize(tensor):
    """CHW normalized tensor -> HWC numpy image in [0, 1], for display."""
    image = tensor.cpu().numpy().transpose(1, 2, 0)
    image = image * STD + MEAN
    return np.clip(image, 0, 1)


def normalize01(arr):
    arr = arr - arr.min()
    return arr / (arr.max() + 1e-8)


def load_model(device):
    model = FlowerCNN().to(device)
    model.load_state_dict(torch.load("flower_cnn.pth", map_location=device))
    model.eval()
    return model


# ----------------------------------------------------------------------
# 1. Training curves
# ----------------------------------------------------------------------

def animate_training_curves(history_path="history.json", save=True):
    path = Path(history_path)
    if not path.exists():
        print(
            f"[curves] {history_path} not found — run the updated train.py "
            "first (it now saves per-epoch history)."
        )
        return

    history = json.loads(path.read_text())
    train_loss = history["train_loss"]
    val_loss = history["val_loss"]
    val_acc = history["val_acc"]
    epochs = list(range(1, len(train_loss) + 1))

    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(11, 4.5))
    fig.suptitle("Training Progress", fontsize=15, fontweight="bold")

    for ax in (ax_loss, ax_acc):
        ax.grid(alpha=0.3, linestyle="--")
        ax.set_xlim(1, max(epochs))
        ax.set_xlabel("Epoch")

    ax_loss.set_ylim(0, max(train_loss + val_loss) * 1.15)
    ax_loss.set_ylabel("Loss")
    ax_loss.set_title("Loss")

    ax_acc.set_ylim(0, 1.05)
    ax_acc.set_ylabel("Accuracy")
    ax_acc.set_title("Validation Accuracy")

    line_train, = ax_loss.plot([], [], color=TEAL, lw=2.5, label="Train")
    line_val, = ax_loss.plot([], [], color=CORAL, lw=2.5, label="Val")
    dot_train, = ax_loss.plot([], [], "o", color=TEAL, ms=8)
    dot_val, = ax_loss.plot([], [], "o", color=CORAL, ms=8)
    ax_loss.legend(facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT)

    line_acc, = ax_acc.plot([], [], color=GOLD, lw=2.5)
    dot_acc, = ax_acc.plot([], [], "o", color=GOLD, ms=8)
    state = {"fill": None}

    def update(frame):
        x = epochs[: frame + 1]
        line_train.set_data(x, train_loss[: frame + 1])
        line_val.set_data(x, val_loss[: frame + 1])
        dot_train.set_data([x[-1]], [train_loss[frame]])
        dot_val.set_data([x[-1]], [val_loss[frame]])

        line_acc.set_data(x, val_acc[: frame + 1])
        dot_acc.set_data([x[-1]], [val_acc[frame]])
        if state["fill"] is not None:
            state["fill"].remove()
        state["fill"] = ax_acc.fill_between(x, val_acc[: frame + 1], color=GOLD, alpha=0.15)

        return line_train, line_val, dot_train, dot_val, line_acc, dot_acc

    anim = animation.FuncAnimation(
        fig, update, frames=len(epochs), interval=400, blit=False, repeat=False
    )
    if save:
        anim.save(OUT_DIR / "training_curves.gif", writer=animation.PillowWriter(fps=3))
        print(f"[curves] saved -> {OUT_DIR / 'training_curves.gif'}")
    plt.tight_layout()
    plt.show()


# ----------------------------------------------------------------------
# 2. Sample predictions grid
# ----------------------------------------------------------------------

def animate_predictions_grid(model, device, n=16, save=True):
    images, labels = next(iter(test_loader))
    images, labels = images[:n].to(device), labels[:n]

    with torch.no_grad():
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)
        preds = probs.argmax(dim=1).cpu()
        confidences = probs.max(dim=1).values.cpu()

    class_names = test_loader.dataset.classes
    cols = 4
    rows = int(np.ceil(n / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.6, rows * 2.9))
    axes = axes.flatten()
    fig.suptitle("Test Predictions", fontsize=15, fontweight="bold")
    for ax in axes:
        ax.axis("off")

    def update(frame):
        ax = axes[frame]
        img = unnormalize(images[frame])
        correct = bool(preds[frame] == labels[frame])
        color = TEAL if correct else CORAL

        ax.imshow(img)
        ax.axis("off")
        rect = plt.Rectangle(
            (-0.5, -0.5), img.shape[1], img.shape[0],
            linewidth=4, edgecolor=color, facecolor="none",
        )
        ax.add_patch(rect)

        label_text = f"{class_names[preds[frame]]} ({confidences[frame] * 100:.0f}%)"
        if not correct:
            label_text += f"\ntrue: {class_names[labels[frame]]}"
        ax.set_title(label_text, fontsize=8.5, color=color, pad=4)
        return (ax,)

    anim = animation.FuncAnimation(
        fig, update, frames=n, interval=350, blit=False, repeat=False
    )
    if save:
        anim.save(OUT_DIR / "predictions_grid.gif", writer=animation.PillowWriter(fps=2.5))
        print(f"[predictions] saved -> {OUT_DIR / 'predictions_grid.gif'}")
    plt.tight_layout()
    plt.show()


# ----------------------------------------------------------------------
# 3. Confusion matrix
# ----------------------------------------------------------------------

def animate_confusion_matrix(model, device, save=True):
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            all_preds.extend(outputs.argmax(dim=1).cpu().tolist())
            all_labels.extend(labels.tolist())

    class_names = test_loader.dataset.classes
    cm = confusion_matrix(all_labels, all_preds)
    n_classes = len(class_names)

    fig, ax = plt.subplots(figsize=(6, 5.5))
    fig.suptitle("Confusion Matrix", fontsize=15, fontweight="bold")

    cmap = LinearSegmentedColormap.from_list("teal_fade", [PANEL, TEAL])
    im = ax.imshow(np.zeros_like(cm), cmap=cmap, vmin=0, vmax=cm.max())
    ax.set_xticks(range(n_classes))
    ax.set_yticks(range(n_classes))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")

    texts = [
        [ax.text(j, i, "", ha="center", va="center", fontsize=10, fontweight="bold")
         for j in range(n_classes)]
        for i in range(n_classes)
    ]
    cells = [(i, j) for i in range(n_classes) for j in range(n_classes)]

    def update(frame):
        display = np.zeros_like(cm)
        for i, j in cells[: frame + 1]:
            display[i, j] = cm[i, j]
            texts[i][j].set_text(str(cm[i, j]))
        im.set_data(display)
        return (im,)

    anim = animation.FuncAnimation(
        fig, update, frames=len(cells), interval=40, blit=False, repeat=False
    )
    if save:
        anim.save(OUT_DIR / "confusion_matrix.gif", writer=animation.PillowWriter(fps=20))
        print(f"[confusion] saved -> {OUT_DIR / 'confusion_matrix.gif'}")
    plt.tight_layout()
    plt.show()


# ----------------------------------------------------------------------
# 4. Conv1 filters + feature maps
# ----------------------------------------------------------------------

def animate_filters_and_features(model, device, save=True):
    conv1 = model.features[0]  # first Conv2d(3 -> 16, kernel_size=3)
    weights = conv1.weight.detach().cpu().numpy()  # (16, 3, 3, 3)

    sample_image, _ = next(iter(test_loader))
    sample_image = sample_image[0:1].to(device)
    with torch.no_grad():
        feature_maps = torch.relu(conv1(sample_image))[0].cpu().numpy()  # (16, 128, 128)

    n_filters = weights.shape[0]
    cols = 8
    rows = 2  # 2 rows of filters, 2 rows of feature maps below them

    fig, axes = plt.subplots(rows * 2, cols, figsize=(cols * 1.5, rows * 3.2))
    fig.suptitle("Conv1 Filters (top) & Feature Maps (bottom)", fontsize=13, fontweight="bold")

    filter_axes = axes[:rows].flatten()
    feature_axes = axes[rows:].flatten()
    for ax in list(filter_axes) + list(feature_axes):
        ax.axis("off")

    def update(frame):
        f_ax = filter_axes[frame]
        m_ax = feature_axes[frame]

        kernel = normalize01(weights[frame].transpose(1, 2, 0))
        f_ax.imshow(kernel)
        f_ax.set_title(f"#{frame}", fontsize=8, color=TEAL)

        fmap = normalize01(feature_maps[frame])
        m_ax.imshow(fmap, cmap="magma")

        return f_ax, m_ax

    anim = animation.FuncAnimation(
        fig, update, frames=n_filters, interval=250, blit=False, repeat=False
    )
    if save:
        anim.save(OUT_DIR / "filters_and_features.gif", writer=animation.PillowWriter(fps=3))
        print(f"[filters] saved -> {OUT_DIR / 'filters_and_features.gif'}")
    plt.tight_layout()
    plt.show()


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Animated visualizations for the Flower CNN.")
    parser.add_argument(
        "--only", nargs="+",
        choices=["curves", "predictions", "confusion", "filters"],
        default=["curves", "predictions", "confusion", "filters"],
        help="Which visualizations to run (default: all).",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    if "curves" in args.only:
        animate_training_curves()

    needs_model = {"predictions", "confusion", "filters"} & set(args.only)
    if needs_model:
        model = load_model(device)
        if "predictions" in args.only:
            animate_predictions_grid(model, device)
        if "confusion" in args.only:
            animate_confusion_matrix(model, device)
        if "filters" in args.only:
            animate_filters_and_features(model, device)


if __name__ == "__main__":
    main()

import torch
import math
import matplotlib.pyplot as plt

from data import test_loader
from model import FlowerCNN

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)


model = FlowerCNN().to(device)

model.load_state_dict(
    torch.load(
        "flower_cnn.pth",
        map_location=device
    )
)

model.eval()


correct = 0
total = 0

all_predictions = []
all_labels = []

misclassified_images = []


# -------------------------
# Test evaluation
# -------------------------

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        predictions = model(images)

        # Predicted class for each image
        predicted_classes = predictions.argmax(dim=1)

        # Collect every prediction and true label
        all_predictions.extend(
            predicted_classes.cpu().tolist()
        )

        all_labels.extend(
            labels.cpu().tolist()
        )

        # Calculate accuracy
        correct += (
            predicted_classes == labels
        ).sum().item()

        total += labels.size(0)

        # Collect misclassified images
        for image, true_label, predicted_label in zip(
            images,
            labels,
            predicted_classes
        ):

            if true_label != predicted_label:

                misclassified_images.append(
                    (
                        image.cpu(),
                        true_label.item(),
                        predicted_label.item()
                    )
                )


# -------------------------
# Test accuracy
# -------------------------

test_accuracy = correct / total

print(
    f"Test Accuracy: {test_accuracy:.4f}"
)

print(
    f"Correct: {correct}/{total}"
)


# -------------------------
# Confusion matrix
# -------------------------

cm = confusion_matrix(
    all_labels,
    all_predictions
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=test_loader.dataset.classes
)

disp.plot(
    xticks_rotation=45
)

plt.tight_layout()
plt.show()

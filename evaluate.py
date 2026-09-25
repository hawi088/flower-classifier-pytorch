import torch
from data import test_loader
from model import FlowerCNN

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)

model = FlowerCNN().to(device)

model.load_state_dict(
    torch.load("flower_cnn.pth",map_location=device)
)

model.eval()
correct = 0
total = 0

for images,labels in test_loader:
    images = images.to(device)
    labels = labels.to(device)

    predictions = model(images)
    predicted_classes = predictions.argmax(dim=1)
    correct +=(
        predicted_classes == labels
    ).sum().item()
    total += labels.size(0)
test_accuracy = correct / total
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Correct: {correct}/{total}")
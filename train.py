import torch
from data import train_loader, val_loader
from model import FlowerCNN

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)

model =  FlowerCNN().to(device)

#loss function
loss_fn   = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001,
)

EPOCHS = 10

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    for images,labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        #forward pass
        predictions = model(images)
        #calculate loss
        loss = loss_fn(predictions,labels)
        #Remove previous gradients
        optimizer.zero_grad()
        #backward propagation
        loss.backward()
        #update parameter
        optimizer.step()
        running_loss +=loss.item()
    epoch_train_loss = running_loss / len(train_loader)
    model.eval()
    val_loss =0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images,labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            predictions = model(images)
            loss = loss_fn(predictions,labels)
            val_loss+=loss.item()
            predicted_classes = predictions.argmax(dim=1)
            correct += (predicted_classes == labels).sum().item()
            total +=labels.size(0)
    epoch_val_loss = val_loss / len(val_loader)
    val_accuracy = correct / total
    print(
        f"Epoch {epoch + 1}/{EPOCHS} | "
        f"Train Loss: {epoch_train_loss:.4f} | "
        f"Val Loss: {epoch_val_loss:.4f} | "
        f"Val Acc: {val_accuracy:.4f}"
    )

torch.save(model.state_dict(), "flower_cnn.pth")
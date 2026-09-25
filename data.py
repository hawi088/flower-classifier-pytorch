import torch
from torchvision.transforms import v2
from torch.utils.data import DataLoader
from torchvision import datasets
#configure
IMAGE_SIZE = 128
BATCH_SIZE = 32

#training transformers 
train_transform = v2.Compose([
    v2.RandomResizedCrop(
        size=(IMAGE_SIZE,IMAGE_SIZE),
        scale=(0.8,1.0),
    ),#take random section of the original image, crop it and resize the cropped image in to 128 x 128, and the scale refers the 80% to 100% of the original image should be present
    v2.RandomHorizontalFlip(), # flip the image right or left randomly (implicitly mentioned 50% of the time)
    v2.ToImage(),
    v2.ToDtype(torch.float32,scale=True)
])

#train transforms 

eval_transform = v2.Compose([
    v2.Resize(
        size=(IMAGE_SIZE,IMAGE_SIZE),
    ),
    v2.ToImage(),
    v2.ToDtype(torch.float32,scale=True),
])

train_dataset = datasets.ImageFolder(
    root="data/train",
    transform=train_transform
)

val_dataset = datasets.ImageFolder(
    root = "data/val",
    transform=eval_transform
)
test_dataset = datasets.ImageFolder(
    root="data/test",
    transform=eval_transform
)


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)
val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False
)
test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)
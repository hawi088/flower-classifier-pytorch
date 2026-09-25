import torch 
import torch.nn as nn
class FlowerCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3,16,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(16,32,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(32,64,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.classfier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64*16*16,5)
        )

    def forward(self,x):
        x = self.features(x)
        x= self.classfier(x)
        return  x
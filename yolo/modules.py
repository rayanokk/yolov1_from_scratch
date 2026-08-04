import torch 
import torch.nn as nn 

class SimpleCNN(nn.Module):
    def __init__(self, in_ch=3, num_class=10):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.relu = nn.ReLU()

        self.fc = nn.Linear(32*8*8, num_class)
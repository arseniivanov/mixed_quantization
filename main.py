import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F

def quantize_int8(x):
    # Scale to [0, 255] then clip and convert to int8
    x = torch.clamp(x, 0, 1)  # Assuming x is normalized between 0 and 1
    return (x * 255).to(torch.int8)

def quantize_int4(x):
    # Scale to [0, 15] then clip and convert to int4-like values within int32
    x = torch.clamp(x, 0, 1)  # Assuming x is normalized between 0 and 1
    return (x * 15).to(torch.int32) & 0xF  # Applying mask to simulate int4

def binarize(x):
    # Binarize to 0 or 1
    return (x > 0.5).to(torch.int32)


class HybridQuantizedNet(nn.Module):
    def __init__(self):
        super(HybridQuantizedNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1, bias=False)  # Example Conv layer
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1, bias=False)
        self.fc1 = nn.Linear(32 * 28 * 28, 120, bias=False)  # Fully connected layers
        self.fc2 = nn.Linear(120, 10, bias=False)

    def forward(self, x):
        x = quantize_int8(F.relu(self.conv1(x)))
        x = quantize_int4(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 28 * 28)  # Flatten
        x = binarize(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Training on {device}")

# Initialize the network
net = HybridQuantizedNet().to(device)

#Optimizer
optimizer = optim.AdamW(net.parameters(), lr=0.001)
loss_function = nn.CrossEntropyLoss()

# Data transformations and dataloader
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=32, shuffle=True)


def train(model, train_loader, optimizer, epochs=5):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)  # Move data to the device
            optimizer.zero_grad()

            outputs = model(images)
            loss = loss_function(outputs, labels)

            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f'Epoch {epoch+1}, Average Loss: {running_loss / len(train_loader)}')

    print('Finished Training')


train(net, trainloader, optimizer)
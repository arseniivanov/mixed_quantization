import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F

def quantize_int8(x):
    x_norm = torch.clamp(x, 0, 1)  # Assuming x is normalized between 0 and 1
    x_quantized = (x_norm * 255).to(torch.int8).float() / 255
    x_ste = x_quantized.detach() - x_norm.detach() + x_norm
    return x_ste

def quantize_int4(x):
    x_norm = torch.clamp(x, 0, 1)  # Assuming x is normalized between 0 and 1
    x_quantized = ((x_norm * 15).to(torch.int8)).float() / 15
    x_ste = x_quantized.detach() - x_norm.detach() + x_norm
    return x_ste    # Scale to [0, 15] then clip and convert to int4-like values within int32


def binarize(x):
    x_binarized = (x > 0.5).to(torch.float)
    x_ste = x_binarized.detach() - x.detach() + x
    return x_ste


class HybridQuantizedNet(nn.Module):
    def __init__(self):
        super(HybridQuantizedNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 128, 3, padding=1, bias=False)  # Example Conv layer
        self.bn1 = nn.BatchNorm2d(128)
        self.conv2 = nn.Conv2d(128, 256, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(256)
        self.fc1 = nn.Linear(256 * 32 * 32, 512, bias=False)  # Fully connected layers
        self.bn3 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 10, bias=False)

    def forward(self, x):
        x = quantize_int8(F.relu(self.bn1(self.conv1(x))))
        x = quantize_int4(F.relu(self.bn2(self.conv2(x))))
        x = x.view(x.size(0), -1)  # Correctly maintain the batch size
        x = binarize(F.relu(self.bn3(self.fc1(x))))
        x = binarize(self.fc2(x))
        return F.log_softmax(x, dim=1)


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


def evaluate_model(model, data_loader, device):
    model.to(device)
    model.eval()  # Set the model to evaluation mode

    correct = 0
    total = 0

    with torch.no_grad():  # No gradients needed for the operation
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    return accuracy


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


train(net, trainloader, optimizer)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = DataLoader(testset, batch_size=32, shuffle=False)


# Evaluate the model
accuracy = evaluate_model(net, testloader, device)
print(f"Test Accuracy: {accuracy:.2f}%")
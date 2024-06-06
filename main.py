import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.ao.quantization as quant
import torch.nn.functional as F
from architectures import FullInt8Quant, BaselineCNN

def train(model, train_loader, optimizer, scheduler, epochs=15):
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

        scheduler.step()
        print(f'Epoch {epoch+1}, Average Loss: {running_loss / len(train_loader)}')

    print('Finished Training')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Training on {device}")

QUANT = True

if QUANT:
    model_fp32 = FullInt8Quant().to(device)
    model_fp32.eval()
    model_fp32.qconfig = quant.get_default_qat_qconfig('x86')
    model_fp32_fused = quant.fuse_modules(model_fp32, [['conv1', 'bn1', 'relu1'], ['conv2', 'bn2', 'relu2']])
    model_fp32_prepared = quant.prepare_qat(model_fp32_fused.train())
else:
    model_fp32_prepared = BaselineCNN().to(device)
#Optimizer
optimizer = optim.AdamW(model_fp32_prepared.parameters(), lr=0.0005)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
loss_function = nn.CrossEntropyLoss()

# Data transformations and dataloader
transform = transforms.Compose([
    transforms.ToTensor(),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=256, shuffle=True)


train(model_fp32_prepared, trainloader, optimizer, scheduler)


model_fp32_prepared = model_fp32_prepared.to('cpu')
model_fp32_prepared.eval()
model_int8 = quant.convert(model_fp32_prepared)

torch.save(model_int8.state_dict(), 'quantized_model_int8.pth')

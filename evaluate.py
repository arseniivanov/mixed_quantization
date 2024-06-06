import torch
import torchvision
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from architectures import FullInt8Quant
import torch.ao.quantization as quant

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

transform = transforms.Compose([
    transforms.ToTensor(),
])
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = DataLoader(testset, batch_size=256, shuffle=False)

model_fp32 = FullInt8Quant()

model_fp32.eval()
# Fuse the modules
model_fp32_fused = quant.fuse_modules(model_fp32, [['conv1', 'bn1', 'relu1'], ['conv2', 'bn2', 'relu2']])

# Prepare the model for quantization
model_fp32_fused.qconfig = quant.get_default_qconfig('fbgemm')
model_fp32_prepared = quant.prepare(model_fp32_fused)

# Convert to quantized model
model_int8 = quant.convert(model_fp32_prepared)

# Load the state dict
quantized_state_dict = torch.load('quantized_model_int8.pth')
model_int8.load_state_dict(quantized_state_dict)

# Set the model to evaluation mode

device = "cpu"
accuracy = evaluate_model(model_int8, testloader, device)
print(f"Test Accuracy: {accuracy:.2f}%")

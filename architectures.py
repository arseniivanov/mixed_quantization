
import torch
import torch.nn as nn
import torch.ao.quantization as quant
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

class FullInt8Quant(nn.Module):
    def __init__(self):
        super(FullInt8Quant, self).__init__()
        self.quant = quant.QuantStub()
        self.conv1 = nn.Conv2d(3, 128, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(128)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(128, 256, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(256)
        self.relu2 = nn.ReLU()
        self.fc1 = nn.Linear(256 * 32 * 32, 512, bias=False)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(512, 10, bias=False)
        self.dequant = quant.DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        x = self.dequant(x)
        return F.log_softmax(x, dim=1)

class HybridQuantizedNet(nn.Module):
    def __init__(self):
        super(HybridQuantizedNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 128, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(128)
        self.conv2 = nn.Conv2d(128, 256, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(256)
        self.fc1 = nn.Linear(256 * 32 * 32, 512, bias=False)
        self.bn3 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 10, bias=False)

    def forward(self, x):
        x = quantize_int8(F.relu(self.bn1(self.conv1(x))))
        x = quantize_int4(F.relu(self.bn2(self.conv2(x))))
        x = x.view(x.size(0), -1)
        x = binarize(F.relu(self.bn3(self.fc1(x))))
        x = binarize(self.fc2(x))
        return F.log_softmax(x, dim=1)


class BaselineCNN(nn.Module):
    def __init__(self):
        super(BaselineCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 128, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(128)
        self.conv2 = nn.Conv2d(128, 256, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(256)
        self.fc1 = nn.Linear(256 * 32 * 32, 512, bias=False)
        self.bn3 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 10, bias=False)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.bn3(self.fc1(x)))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

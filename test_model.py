import torch
from model import CustomCNN


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model = CustomCNN().to(device)

x = torch.randn(
    32, 1, 224, 224
).to(device)

output = model(x)

print("Input shape:", x.shape)
print("Output shape:", output.shape)
print("Device:", output.device)
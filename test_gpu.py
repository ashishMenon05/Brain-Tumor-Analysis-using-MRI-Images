import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", device)
print("GPU:", torch.cuda.get_device_name(0))

x = torch.randn(5000, 5000).to(device)
y = torch.randn(5000, 5000).to(device)

z = x @ y

print("Tensor device:", z.device)
print("GPU test successful!")
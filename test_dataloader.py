from dataset import BrainTumorDataset
from torch.utils.data import DataLoader


train_dataset = BrainTumorDataset("train.csv")

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)


images, labels = next(iter(train_loader))


print("Image batch shape:", images.shape)
print("Label batch shape:", labels.shape)

print("Labels:", labels)
print("Image minimum:", images.min().item())
print("Image maximum:", images.max().item())
import torch
from model import CNNmodel
from dataset import train_loader, test_loader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = CNNmodel().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss()

# ===== TRAIN =====
for epoch in range(3):
    model.train()
    total_loss = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

    model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print("Accuracy:", correct / total)

feature_maps = {}

def hook_fn(name):
    def hook(module, input, output):
        feature_maps[name] = output.detach()
    return hook

model.conv1.register_forward_hook(hook_fn("conv1"))
model.conv2.register_forward_hook(hook_fn("conv2"))

images, _ = next(iter(train_loader))
images = images.to(device)

_ = model(images)

print(feature_maps["conv1"].shape)
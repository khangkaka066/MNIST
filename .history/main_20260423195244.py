import torch

from model import CNNmodel
from dataset import train_loader, test_loader
from train import train, test


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CNNmodel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()

    epochs = 5

    for epoch in range(epochs):
        loss = train(model, device, train_loader, optimizer, criterion)
        acc = test(model, device, test_loader)

        print(f"Epoch {epoch+1}: Loss = {loss:.4f}, Accuracy = {acc:.4f}")

    # lưu model
    torch.save(model.state_dict(), "mnist_cnn.pth")


if __name__ == "__main__":
    main()
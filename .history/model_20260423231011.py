import torch

class CNNmodel(torch.nn.Module):
    def __init__(self):
        super(CNNmodel, self).__init__()

        self.conv1 = torch.nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = torch.nn.Conv2d(64, 128, kernel_size=3, padding=1)

        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        # 28 → 14 → 7 → 3
        self.fc1 = torch.nn.Linear(128 * 3 * 3, 128)
        self.fc2 = torch.nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(torch.nn.functional.relu(self.conv1(x)))  # 28 → 14
        x = self.pool(torch.nn.functional.relu(self.conv2(x)))  # 14 → 7
        x = self.pool(torch.nn.functional.relu(self.conv3(x)))  # 7 → 3

        x = x.view(x.size(0), -1)

        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.dropout(x, training=self.training)
        x = self.fc2(x)

        return x
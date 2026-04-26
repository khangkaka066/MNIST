import torch
import torch.nn.functional as F

class CNNmodel(torch.nn.Module):
    def __init__(self, num_classes=10, dropout_rate=0.3, input_channel=1):
        super(CNNmodel, self).__init__()

        self.conv1 = torch.nn.Conv2d(1, 64, 3, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(64)

        self.conv2 = torch.nn.Conv2d(64, 128, 3, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(128)

        self.conv3 = torch.nn.Conv2d(128, 256, 3, padding=1)
        self.bn3 = torch.nn.BatchNorm2d(256)

        self.pool = torch.nn.MaxPool2d(2, 2)

        self.fc1 = torch.nn.Linear(256 * 3 * 3, 128)
        self.fc2 = torch.nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))

        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.3, training=self.training)

        x = self.fc2(x)
        return x
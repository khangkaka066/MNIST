import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNmodel(torch.nn.Module):
    def __init__(self, num_classes=10, dropout_rate=0.3, input_channels=1):
        super(CNNmodel, self).__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.block1 = self._make_separable_block(32, 64, stride=2)
        self.block2 = self._make_separable_block(64, 128, stride=2)
        self.block3 = self._make_separable_block(128, 256, stride=2)

        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(256, 128)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))

        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.3, training=self.training)

        x = self.fc2(x)
        return x
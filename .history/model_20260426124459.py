import torch
import torch.nn.functional as F

class CNNmodel(torch.nn.Module):
    def __init__(self, num_classes=10, dropout_rate=0.3, input_channel=1):
        super(CNNmodel, self).__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))

        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.3, training=self.training)

        x = self.fc2(x)
        return x
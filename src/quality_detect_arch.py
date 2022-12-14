import torch
import torch.nn.functional as F
import torch.nn as nn


class QualityDetectNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.dpo = nn.Dropout(p=0.2)

        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=16, kernel_size=5, padding="same"
        )
        self.conv2 = nn.Conv2d(
            in_channels=16, out_channels=32, kernel_size=5, padding="same"
        )
        self.conv3 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=5, padding="same"
        )
        self.conv4 = nn.Conv2d(
            in_channels=64, out_channels=128, kernel_size=5, padding="same"
        )

        self.fc1 = nn.Linear(4 * 16 * 128, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))

        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = self.dpo(x)
        x = F.relu(self.fc1(x))
        x = self.dpo(x)
        x = F.relu(self.fc2(x))
        x = self.dpo(x)
        x = self.fc3(x)
        return x

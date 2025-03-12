import torch
import torch.nn as nn
import torch.optim as optim

class MancalaCNN(nn.Module):

    def __init__(self):
        super().__init__()
        # Input: 2x6 grid for each player
        self.conv1 = nn.Conv2d(2, 16, kernel_size=3, stride=1, padding=1)
        self.bnd1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.bnd2 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.relu = nn.ReLU(inplace=True)
        # self.deconv1 = nn.ConvTranspose2d(32, 32, kernel_size=3, stride=1, padding=1, output_padding=1)
        # self.bn1 = nn.BatchNorm2d(32)
        # self.deconv2 = nn.ConvTranspose2d(32, 16, kernel_size=3, stride=1, padding=1, output_padding=1)
        # self.bn2 = nn.BatchNorm2d(16)
        # self.classifier = nn.Conv2d(16, 6, kernel_size=1)
        self.fc1 = nn.Linear(32 * 1 * 6, 128)
        self.fc2 = nn.Linear(128, 6)

    def forward(self, x):
        x1 = self.bnd1(self.relu(self.conv1(x)))
        x2 = self.bnd2(self.relu(self.conv2(x1)))
        x2 = x2.view(x2.size(0), -1)

        y1 = self.relu(self.fc1(x2))
        y2 = self.fc2(y1)

        # y1 = self.bn1(self.relu(self.deconv1(x2)))
        # y2 = self.bn2(self.relu(self.deconv2(y1)))

        # score = self.classifier(y2)

        return y2

model = MancalaCNN()

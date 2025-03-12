import torch
import torch.nn as nn

class MancalaCNN(nn.Module):

    def __init__(self):
        super().__init__()
        # Input: 2x6 grid for each player + 1 extra channel for player turn information
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.bnd1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.bnd2 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.relu = nn.ReLU(inplace=True)

        self.fc1 = nn.Linear(32 * 1 * 6, 128)
        # self.fc2 = nn.Linear(128, 12)
        
        self.policy_head = nn.Linear(128, 12)  
        self.value_head = nn.Linear(128, 1) 

    def forward(self, x):
        x1 = self.bnd1(self.relu(self.conv1(x)))
        x2 = self.bnd2(self.relu(self.conv2(x1)))

        x2 = x2.view(x2.size(0), -1)

        y1 = self.relu(self.fc1(x2))
        # y2 = self.fc2(y1)

        move_probs = self.policy_head(y1)
        move_probs = torch.softmax(move_probs, dim=-1)

        state_value = self.value_head(y1)
        state_value = torch.tanh(state_value)

        return move_probs, state_value

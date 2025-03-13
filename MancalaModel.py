import torch
import torch.nn as nn

class MancalaModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.relu = nn.ReLU(inplace=True)

        self.lin1 = nn.Linear(13, 128)
        self.lin2 = nn.Linear(128, 128)
        
        self.policy_head = nn.Linear(128, 12)  
        self.value_head = nn.Linear(128, 1) 

    def forward(self, x):
        x1 = self.relu(self.lin1(x))
        x2 = self.relu(self.lin2(x1))

        move_probs = self.policy_head(x2)
        move_probs = torch.softmax(move_probs, dim=-1)

        state_value = self.value_head(x2)
        state_value = torch.tanh(state_value)

        return move_probs, state_value

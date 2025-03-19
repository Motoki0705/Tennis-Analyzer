import torch
import torch.nn as nn

class ChildModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.fc_in = nn.Linear(28 * 28, 28 * 16)
        self.norm_in = nn.BatchNorm1d(28 * 16)
        self.ac_in = nn.ReLU()
        self.fc_middle = nn.Linear(28 * 16, 28 * 4)
        self.norm_middle = nn.BatchNorm1d(28 * 4)
        self.ac_middle = nn.ReLU()
        self.fc_out = nn.Linear(28 * 4, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc_in(x)
        x = self.norm_in(x)
        x = self.ac_in(x)
        x = self.fc_middle(x)
        x = self.norm_middle(x)
        x = self.ac_middle(x)
        x = self.fc_out(x)
        return x
    

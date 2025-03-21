import torch
import torch.nn as nn

class ChildModel(nn.Module):
    def __init__(self, num_classes, input_shape):
        super().__init__()
        h, w = input_shape
        input_square = h * w
        self.fc_in = nn.Linear(input_square, input_square // 2)
        self.norm_in = nn.BatchNorm1d(input_square // 2)
        self.ac_in = nn.ReLU()
        self.fc_middle = nn.Linear(input_square // 2, input_square // 4)
        self.norm_middle = nn.BatchNorm1d(input_square // 4)
        self.ac_middle = nn.ReLU()
        self.fc_out = nn.Linear(input_square // 4, num_classes)

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
    

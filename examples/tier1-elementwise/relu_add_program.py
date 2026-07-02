import torch
import torch.nn.functional as F


class ReluAddModule(torch.nn.Module):
    def forward(self, x, y):
        return F.relu(x + y)

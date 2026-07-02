import torch
import torch.nn.functional as F


class GeluModule(torch.nn.Module):
    def forward(self, x):
        return F.gelu(x)

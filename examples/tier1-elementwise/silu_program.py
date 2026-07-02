import torch
import torch.nn.functional as F


class SiluModule(torch.nn.Module):
    def forward(self, x):
        return F.silu(x)

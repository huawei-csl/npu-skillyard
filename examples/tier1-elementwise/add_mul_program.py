import torch


class AddMulModule(torch.nn.Module):
    def forward(self, x, bias, scale):
        return (x + bias) * scale

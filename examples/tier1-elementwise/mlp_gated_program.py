import torch


class TinyGatedMLPModule(torch.nn.Module):
    def forward(self, gate, up):
        return torch.nn.functional.silu(gate) * up

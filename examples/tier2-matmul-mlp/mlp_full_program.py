import torch
import torch.nn.functional as F


class Projection(torch.nn.Module):
    def forward(self, x, weight):
        return F.linear(x, weight)


class MLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = Projection()
        self.up_proj = Projection()
        self.down_proj = Projection()
        self.act_fn = torch.nn.SiLU()

    def forward(self, x, gate_w, up_w, down_w):
        gate = self.gate_proj(x, gate_w)
        up = self.up_proj(x, up_w)
        gated = self.act_fn(gate) * up
        return self.down_proj(gated, down_w)


class TinyFullMLPModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = MLP()

    def forward(self, x, gate_w, up_w, down_w):
        return self.mlp(x, gate_w, up_w, down_w)

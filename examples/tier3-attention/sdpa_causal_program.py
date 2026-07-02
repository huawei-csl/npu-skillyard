import torch
import torch.nn.functional as F


class SdpaCausalModule(torch.nn.Module):
    def forward(self, q, k, v):
        return F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=True)

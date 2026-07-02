import torch
import torch.nn.functional as F


class SdpaModule(torch.nn.Module):
    def forward(self, q, k, v):
        return F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=False,
        )

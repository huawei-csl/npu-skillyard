import torch
import torch.nn.functional as F


class Projection(torch.nn.Module):
    def forward(self, x, weight):
        return F.linear(x, weight)


class SelfAttention(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = Projection()
        self.k_proj = Projection()
        self.v_proj = Projection()
        self.o_proj = Projection()
        self.num_heads = 4
        self.head_dim = 32

    def _split_heads(self, x):
        batch, seq_len, _hidden = x.shape
        return x.reshape(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

    def _merge_heads(self, x):
        batch, _heads, seq_len, _head_dim = x.shape
        return x.transpose(1, 2).reshape(batch, seq_len, self.num_heads * self.head_dim)

    def forward(self, x, q_w, k_w, v_w, o_w):
        q = self._split_heads(self.q_proj(x, q_w))
        k = self._split_heads(self.k_proj(x, k_w))
        v = self._split_heads(self.v_proj(x, v_w))
        attn = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=False,
        )
        return self.o_proj(self._merge_heads(attn), o_w)


class TinyAttentionBlockModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = SelfAttention()

    def forward(self, x, q_w, k_w, v_w, o_w):
        return self.self_attn(x, q_w, k_w, v_w, o_w)

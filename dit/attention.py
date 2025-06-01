import torch
from torch import nn
from torch.nn import functional as f
import math

class SelfAttention(nn.Module):

    def __init__(self, n_heads: int, d_embed: int, in_proj_bias=True, out_proj_bias=True):
        super().__init__()

        self.in_proj = nn.Linear(d_embed, 3 * d_embed, bias=in_proj_bias)
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)
        self.n_heads = n_heads
        self.d_head = d_embed // n_heads

    def forward(self, x: torch.Tensor, causal_mask=False):
        # x: (Batch_Size, Seq_Len, Dim)

        input_shape = x.shape

        batch_size, sequence_length, d_embed = input_shape

        intermin_shape = (batch_size, sequence_length, self.n_heads, self.d_head)

        # (B, Seq_Len, Dim) -> (B, Seq_Len, Dim * 3) 
        # -> 3 tensors of shape (B, Seq_Len, Dim)
        q, k, v = self.in_proj(x).chunk(3, dim=-1) 
        # (Batch_Size, Seq_Len, Dim) -> (B, Seq_Len, H, Dim / H) -> (B, H, Seq_Len, Dim/H)
        q = q.view(intermin_shape).transpose(1,2)
        k = k.view(intermin_shape).transpose(1,2)
        v = v.view(intermin_shape).transpose(1,2)
        # (B, H, Seq_Len, Seq_Len) (T x T attention map)
        weight = q @ k.transpose(-1,-2)

        if causal_mask:
            # Mask where the upper triangular is made of 1s
            mask = torch.ones_like(weight, dtype=torch.bool).triu(1)
            weight.masked_fill_(mask, -torch.inf)

        weight /= math.sqrt(self.d_head)

        weight = f.softmax(weight, dim=-1)

        # (B, H, Seq_Len, Seq_Len) @ (B, H, Seq_Len, Dim/H) -> (B, H, Seq_Len, Dim/H)
        output = weight @ v
        # (B, H, Seq_Len, Dim/H) -> (B, Seq_Len, H, Dim/H)
        output = output.transpose(1,2)

        output = output.reshape(input_shape)

        output = self.out_proj(output)
        # (Batch_Size, Seq_Len, Dim)
        return output
        
class CrossAttention(nn.Module): #similar to above selfattentionblock

    def __init__(self, n_heads: int, d_embed: int, d_cross: int, in_proj_bias=True, out_proj_bias=True):
        super().__init__()
        self.q_proj = nn.Linear(d_embed, d_embed, bias=in_proj_bias)
        self.k_proj = nn.Linear(d_cross, d_cross, bias=in_proj_bias)
        self.v_proj = nn.Linear(d_cross, d_cross, bias=in_proj_bias)
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)
        self.n_heads = n_heads
        self.d_head = d_embed // n_heads

    def forward(self, x, y):
        # x: (latent): (B, Seq_Len_Q, Dim_Q)
        # y: (context): (B, Seq_Len_KV, Dim_KV) = (B, 77, 768)

        input_shape = x.shape

        batch_size, sequence_length, d_embed = input

        interim_shape = (batch_size, -1, self.n_heads, self.d_heads)

        # Multiply query by Wq
        q = self.q_proj(x) # (B, Seq_Q, d_embed)
        k = self.k_proj(y) # (B, Seq_KV, d_cross)
        v = self.v_proj(y) # (B, Seq_KV, d_cross)
        

        q = q.view(interim_shape).transpose(1,2) # (B, n_heads, SeqQ, d_heads)
        k = k.view(interim_shape).transpose(1,2) # (B, n_heads, SeqKV, d_heads)
        v = v.view(interim_shape).transpose(1,2) # (B, n_heads, SeqKV, d_heads)

        # attention formula
        weight = q @ k.transpose(-1,-2) # -> (B, n_heads, SeqQ, SeqKV)

        weight /= math.sqrt(self.d_head)

        weight = f.softmax(weight, dim=-1)

        output = weight @ v # (B, n_heads, SeqQ, d_heads)

        output = output.transpose(1, 2).continuous() # -> (B, SeqQ, n_heads, d_head)

        output = output.view(input_shape) # -> (B, SeqQ, d_embed)

        output = self.out_proj(output)

        return output # (B, Seq_Len_Q, Dim_Q) same as d_embed
    



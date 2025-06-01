import torch
from torch import nn
from torch.nn import functional as F
import math

# replicating DiT-S, 12 layers, d=384, 6 heads
# assume patch size, p = 8, thus input is 16 patches
# input to block is (B, 16, 384), along with timestep and class emb

#timestep + class, are both (B, 384)

class CondMLP(nn.Module):
    # returns gamma, beta, alphas for AdaLNZ + scale, shift
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(256, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.linear = nn.Linear(hidden_dim, 6 * hidden_dim)
        # AdaLNZero, initializing alpha1, alpha2 = 0
        with torch.no_grad():
            start = 4 * hidden_dim
            end = 6 * hidden_dim

            self.linear.weight[start:end].zero_()
            self.linear.bias[start:end].zero_()


    
    def forward(self, t_emb, c_emb):
        # t_emb = (B, 256)
        # c_emb = (B, D) = (B, 384)
        t_emb = self.mlp(t_emb) # (B, 256) -> (B, D)
        cond = t_emb + c_emb # (B, D)
        cond = F.silu(cond) # (B, D)
        cond = self.linear(cond) # (B, 6D)
        g1, b1, g2, b2, a1, a2 = cond.chunk(6, dim=-1) # each (B, D)
        return g1, b1, g2, b2, a1, a2


class AdaLN(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)


    def forward(self, x, gamma, beta):
        # x: (B, T, D)
        # gamma & beta: (B, D)
        x = self.norm(x) # (B, 16, 384)
        # g, b become (B, 1, 384)
        x = x * gamma.unsqueeze(1) + beta.unsqueeze(1)
        return x # (B, 16, 384)

class FeedForward(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.linear1 = nn.Linear(hidden_dim, 4 * hidden_dim)
        self.dropout = nn.Dropout(0.1)
        self.linear2 = nn.Linear(4 * hidden_dim, hidden_dim)
    
    def gelu(self, x):
        return 0.5 * x * (1 + torch.tanh(((2.0 / math.pi)**0.5) * (x + 0.044715*x.pow(3))))

    def forward(self, x):
        # x: (B, 16, 384)
        x = self.linear1(x)
        x = self.gelu(x)
        #x = self.dropout(x)
        x = self.linear2(x)
        #x = self.dropout(x)
        return x

class MultiHeadSelfAttention(nn.Module): # DiT-S uses 6 heads
    def __init__(self, n_heads: int, d_embed: int, in_proj_bias=True, out_proj_bias=True):
        super().__init__()

        self.in_proj = nn.Linear(d_embed, 3 * d_embed, bias=in_proj_bias)
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)
        self.n_heads = n_heads
        assert d_embed % n_heads == 0
        self.d_head = d_embed // n_heads


    def forward(self, x: torch.Tensor):
        # Input (B, 16, D) = (B, 16, 384)
        input_shape = x.shape

        batch_size, num_patches, d_embed = input_shape

        intermediate_shape = (batch_size, num_patches, self.n_heads, self.d_head)

        # (B, T, Dim) -> (B, T, Dim * 3) 
        q, k, v = self.in_proj(x).chunk(3, dim=-1)

        # (B, T, D) -> (B, H, T, Dim / H)
        q = q.view(intermediate_shape).transpose(1,2)
        k = k.view(intermediate_shape).transpose(1,2)
        v = v.view(intermediate_shape).transpose(1,2)
        # (B, H, T, T), T x T attention map
        weight = q @ k.transpose(-1,-2)
        weight /= math.sqrt(self.d_head)
        weight = F.softmax(weight, dim=-1)
        # results in (B, H, T, Dim/H)
        output = weight @ v
        # (B, T, H, Dim/H)
        output = output.transpose(1,2)
        # back to (B, T, D)
        output = output.reshape(input_shape)
        output = self.out_proj(output)
        return output # (B, 16, 384)

class DiTBlock(nn.Module):
    def __init__(self, hidden_dim: int, n_heads: int, ):
        super().__init__()

        self.cond = CondMLP(hidden_dim)
        self.adalnzero = AdaLN(hidden_dim)
        self.mhsa = MultiHeadSelfAttention(n_heads=n_heads, d_embed=hidden_dim)
        self.ffw = FeedForward(hidden_dim)

    def forward(self, x: torch.Tensor, t_embd: torch.Tensor, c_embd: torch.Tensor):
        # t_emb = (B, 256)
        # c_emb = (B, D) = (B, 384)
        # x: (B, patch_size, C) -> (B, 16, 384)
        residual1 = x
        gamma1, beta1, gamma2, beta2, alpha1, alpha2 = self.cond(t_embd, c_embd)

        x = self.adalnzero(x, gamma1, beta1)
        x = self.mhsa(x)
        x = x * alpha1.unsqueeze(1)
        x = x + residual1

        residual2 = x
        x = self.adalnzero(x, gamma2, beta2)
        x = self.ffw(x)
        x = x * alpha2.unsqueeze(1)
        x = x + residual2
        return x #output (B, )


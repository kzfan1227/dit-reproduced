import torch
from torch import nn
from torch.nn import functional as F
import math
from dit.dit_block import DiTBlock, AdaLN, FeedForward

class TimeEmbedding(nn.Module):
    # input : (B, 256)
    # turns freq-time embedding into (B, 256)

     def __init__(self, orig_dim: int):
         super().__init__()

         self.linear1 = nn.Linear(orig_dim, 256)
         self.linear2 = nn.Linear(256, 256)


     def forward(self, x):
         x = self.linear1(x)
         x = F.silu(x)
         x = self.linear2(x)
         return x


class ClassEmbedding(nn.Module):
    # input (B, 1000)
    # turns classes into (B, 384) embd
     def __init__(self, orig_dim: int, hidden_dim: int):
        super().__init__()
        self.linear1 = nn.Linear(orig_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
    
     def forward(self, classes):
        classes = self.linear1(classes)
        classes = F.silu(classes)
        classes = self.linear2(classes)
        return classes # (B, 384)
     

class Patchify(nn.Module):
    # Patchify 32x32x4 latent into sequence of d embeddings
    # Add sin/cos positional embeddings to all tokens
    # (B, C, I, I) -> (B, (I/p)^2, D)
    # (B, 4, 32, 32) -> (B, 16, 384)
    # p = 8
    def __init__(self, latent_size: int, hidden_dim: int,
                 patch_size: int, num_channels: int):
        super().__init__()
        self.latent_size = latent_size
        self.hidden_dim = hidden_dim
        self.patch_size = patch_size
        self.num_channels = num_channels

        self.num_patches = (self.latent_size // self.patch_size) ** 2
        
        # (B, C, I, I) -> (B, D, I/p, I/p)
        # (B, 4, 32, 32) -> (B, 384, 4, 4)
        self.patch_proj = nn.Conv2d(self.num_channels, self.hidden_dim, 
                                    kernel_size=self.patch_size, stride=self.patch_size)
        
    def sinusoidal_positional_emb(self, length, dim):

        positions_matrix = torch.zeros(length, dim)
        pos = torch.arange(0, length).unsqueeze(1) # (16, 1)

        # P(k, 2i) = sin( k * 1 / 10000^(2i/d))
        # P(k, 2i + 1) = cos (K * 1 / 10000^(2i/d))
        div_term = torch.exp((torch.arange(0, dim, 2, dtype=torch.float) *
                         -(math.log(10000.0) / dim)))
        
        positions_matrix[:, 0::2] = torch.sin(pos.float() * div_term)
        positions_matrix[:, 1::2] = torch.cos(pos.float() * div_term)
        # (16, 384)
        return positions_matrix

    def forward(self, x):
        # x: (B, C, I, I) = (B, 4, 32, 32)
        # (B, 4, 32, 32) -> (B, 384, 4, 4)
        x = self.patch_proj(x)
        # (B, 384, 4, 4) -> (B, 384, 16)
        x = x.flatten(2)
        # (B, 384, 16) -> (B, 16, 384)
        x = x.transpose(-1, -2)
        # (B, 16, 384)
        x = x + self.sinusoidal_positional_emb(self.num_patches, self.hidden_dim)
        return x

class OutputCondMLP(nn.Module):
    # returns gamma, beta for final AdaLN, shift
    # input (B, 16, 384)
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(256, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.linear = nn.Linear(hidden_dim, 2 * hidden_dim)

    
    def forward(self, t_emb, c_emb):
        # t_emb = (B, 256)
        # c_emb = (B, D) = (B, 384)
        t_emb = self.mlp(t_emb) # (B, 256) -> (B, D)
        cond = t_emb + c_emb # (B, D)
        cond = F.silu(cond) # (B, D)
        cond = self.linear(cond) # (B, 2D)
        g, b = cond.chunk(2, dim=-1) # each (B, D)
        return g, b


class DiffusionTransformer(nn.Module):
    # latent, time embedder, class embedder, hiddendim, patchsize, num_heads
    def __init__(self, patch_size: int, num_attention_heads: int, num_blocks: int, hidden_dim=384, num_channels=4, 
                 latent_size=32):
        super().__init__()
        self.num_blocks = num_blocks
        self.num_channels = num_channels
        self.latent_size = latent_size
        self.hidden_dim = hidden_dim
        self.patch_size = patch_size
        self.num_attention_heads = num_attention_heads

        self.embed_time = TimeEmbedding(256)
        self.embed_class = ClassEmbedding(1000, self.hidden_dim)
        self.patchify = Patchify(self.latent_size, self.hidden_dim, self.patch_size, self.num_channels)
        self.ditblocks = nn.ModuleList([DiTBlock(self.hidden_dim, self.num_attention_heads) for _ in range(self.num_blocks)])
        self.final_ln_cond = OutputCondMLP(self.hidden_dim)
        self.final_ln = AdaLN(self.hidden_dim)
        # (B, N, D) (B, 16, 384) -> (B, N, P*P*2C) (B, 16, 8*8*4*2)
        self.output_proj = nn.Linear(self.hidden_dim, 2 * self.patch_size * self.patch_size * self.num_channels)


    
    def forward(self, latent: torch.Tensor, time_tensor: torch.Tensor,
                 class_tensor: torch.Tensor):
        # input a (B x 4 x 32 x 32) latent
        # (B, ?) time and (B, ~1000) class tensors
        time_embd = self.embed_time(time_tensor)
        class_embd = self.embed_class(class_tensor)

        x = self.patchify(latent)
        
        for block in self.ditblocks:
            x = block(x, time_embd, class_embd)
        
        gamma, beta = self.final_ln_cond(time_embd, class_embd)
        x = self.final_ln(x, gamma, beta)
        # (B, N, D) (B, 16, 384) -> (B, N, P*P*2C) | (B, I^2/p^2, p^2 * 2c) (B, 16, 8*8*4*2) 
        x = self.output_proj(x)

        B, N, patch_dim = x.shape
        Ip = int(N ** 0.5)
        p = self.patch_size
        # (B, 16, 8*8*4*2) -> (B, 4, 32, 32)

        # (B, I^2/p^2, 2*p*p*c) -> (B, I/p, I/p, p, p, 2c)
        x = x.reshape(B, Ip, Ip, p, p, 2*self.num_channels)
        # (B, I/p, I/p, p, p, 2c) -> (B, 2c, I/p, p, I/p, p)
        x = x.permute(0, 5, 1, 3, 2, 4)
        # (B, 2c, I/p, p, I/p, p) -> (B, 2C, I, I)
        x = x.reshape(B, 2*self.num_channels, Ip*p, Ip*p)

        noise, cov_matrix = x.chunk(2, dim=1)
        return noise, cov_matrix


class DiffusionTransformerModel(nn.Module):
    def __init__(self, dit):
        super().__init__()
        self.dit = dit
    
    def forward(self, x, t, y=None):
        t = t.float()

        if y is None:
            y = torch.zeros(x.shape[0], 1000)
        elif y.dim() == 1:
            y = F.one_hot(y, num_classes=1000).float()
        
        noise, cov = self.dit(x, t, y)
        return noise
    
# - set up VAE enc/dec code and weights
# - assemble pipeline, figure out time/class tensors (i.e get_time_emb)
# setup imagenet, training loop, sampling

            



        
    




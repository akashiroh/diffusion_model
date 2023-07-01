
import torch
import torch.nn as nn
import torch.nn.functional as F

import einops

from timm.models.layers import DropPath

# --- Utility Functions

def split_heads(x, num_heads):
    return einops.rearrange(x, 'b t (h f) -> (b h) t f', h=num_heads)

def join_heads(x, num_heads):
    return einops.rearrange(x, '(b h) t f -> b t (h f)', h=num_heads)

def attention(query, key, value, mask: torch.Tensor | None = None, flash=False):
        if flash:
            x = nn.functional.scaled_dot_product_attention(query, key, value)
        else:
            scale = 1 / query.shape[-1] ** 0.5
            query = query * scale
            attn = query @ key.transpose(-2, -1)
            if mask is not None:
                attn = attn + mask
            attn = attn.softmax(-1)
            x = attn @ value

        return x

# --- Layers & Blocks

class MLP(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()

        self.layers = nn.Sequential(
                nn.Linear(embed_dim, 4 * embed_dim),
                nn.GELU(),
                nn.Linear(4 * embed_dim, embed_dim),
                )

    def forward(self, x):
        x = self.layers(x)
        return x

class ConvNeXtBlock(nn.Module):
    def __init__(self, input_channels: int, init_layer_scale_val: float =0.0, drop_rate: float = 0.0):
        super().__init__()
        self.depthconv = nn.Conv2d(input_channels, input_channels, kernel_size=7, padding=3, groups=input_channels)
        self.norm = nn.GroupNorm(num_groups=1, num_channels=input_channels, eps=1e-6)
        self.pwconv1 = nn.Linear(input_channels, 4 * input_channels)
        self.activation = nn.GELU()
        self.pwconv2 = nn.Linear(4 * input_channels, input_channels)
        self.gamma = nn.Parameter(init_layer_scale_val * torch.ones(input_channels), requires_grad=True) if init_layer_scale_val > 0 else None
        self.drop_path = DropPath(drop_rate) if drop_rate else None

    def forward(self, x: torch.Tensor):
        # Store residual connection
        residual = x
        
        # Apply depthwise convolution
        x = self.depthconv(x)
        x = self.norm(x)
        # x = x.permute(0, 2, 3, 1)

        # Rearrange so channels are the first dim for the pointwise convolutions
        x = einops.rearrange(x, 'b c h w -> b h w c')

        x = self.pwconv1(x)
        x = self.activation(x)
        x = self.pwconv2(x)
        if self.gamma:
            x = self.gamma * x

        # Rearrange tensor to its original dimensions
        x = einops.rearrange(x, 'b h w c -> b c h w')
        # x = x.permute(0, 3, 1, 2)

        # Apply residual connection
        x = residual + x
        return x

class SelfAttentionBlock(nn.Module):
    def __init__(self, in_dim, embed_dim: int = 2048, num_heads: int = 16):
        super().__init__()

        self.flash = self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")

        self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.in_proj = nn.Linear(in_dim, embed_dim * 3)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.atnorm = nn.LayerNorm(embed_dim)
        self.mlpnorm = nn.LayerNorm(embed_dim)

        self.mlp = MLP(embed_dim)

    def forward(self, x, mask: torch.Tensor | None = None):
        
        # Store residual value
        res = x
        
        x = self.atnorm(x)

        # Project input to query, key, and value
        query, key, value = self.in_proj(x).split(self.embed_dim, dim=2)
        # Rearrange the tensors for multiheaded attention calculations
        query, key, value = [split_heads(t, self.num_heads) for t in [query, key, value]]
        
        # Calculate self attention
        x = attention(query, key, value, mask=mask, flash=self.flash)

        # Rearrange the tensors to their original shapes after multihead attention
        x = join_heads(x, self.num_heads)

        x = self.out_proj(x)
        
        # Apply the residual connection and store another
        x = res + x

        res = x

        # Feed through MLP and apply the residual connection
        x = self.mlpnorm(x)
        x = self.mlp(x)

        x = res + x

        return x

class CrossAttentionBlock(nn.Module):
    def __init__(self, q_dim: int, kv_dim: int, embed_dim: int = 2048, num_heads: int = 16, ff_dim: int = 1024):
        super().__init__()        

        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")

        self.embed_dim = embed_dim

        self.q_proj = nn.Linear(q_dim, embed_dim)
        self.kv_proj = nn.Linear(kv_dim, embed_dim * 2)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.atnorm = nn.LayerNorm(embed_dim)
        self.mlpnorm = nn.LayerNorm(embed_dim)

        self.mlp = MLP(embed_dim)
        
        self.ff = nn.Sequential(
                nn.Linear(kv_dim, ff_dim),
                nn.GELU(),
                nn.Linear(ff_dim, kv_dim),
                )
        self.ffnorm = nn.LayerNorm(kv_dim)

    def forward(self, q: torch.Tensor, kv: torch.Tensor, mask: torch.Tensor | None = None):
        query = self.q_proj(q)
        key, value = self.kv_proj(kv).split(self.embed_dim, dim=2)

        x = attention(query, key, value, mask=mask, flash=self.flash)
        x = self.out_proj(x)
        x = self.atnorm(x)

        x = self.mlp(x)
        x = self.mlpnorm(x)

        return x





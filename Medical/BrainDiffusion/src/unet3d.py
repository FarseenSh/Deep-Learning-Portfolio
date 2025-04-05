"""
3D UNet architecture for brain diffusion models.

This module implements memory-efficient 3D UNet with attention mechanisms
specifically designed for diffusion models of volumetric brain data.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

class SinusoidalPosEmb(nn.Module):
    """Sinusoidal timestep embedding."""
    
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        
    def forward(self, x):
        """
        Args:
            x: Timestep tensor [B]
            
        Returns:
            Timestep embeddings [B, dim]
        """
        half_dim = self.dim // 2
        emb = torch.log(torch.tensor(10000.0, device=x.device)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=x.device) * -emb)
        emb = x.unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat((torch.sin(emb), torch.cos(emb)), dim=1)
        return emb


class EfficientAttention3D(nn.Module):
    """Memory-efficient attention for 3D volumes using linear attention."""
    
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5
        inner_dim = heads * dim_head
        
        self.to_qkv = nn.Conv3d(dim, inner_dim * 3, 1, bias=False)
        self.to_out = nn.Conv3d(inner_dim, dim, 1)
        
    def forward(self, x):
        """
        Args:
            x: Input feature volume [B, C, D, H, W]
            
        Returns:
            Attention-enhanced features
        """
        b, c, d, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: t.reshape(b, self.heads, self.dim_head, d, h, w), qkv)
        
        # Linear attention - O(n) instead of O(n²)
        q = torch.nn.functional.softmax(q, dim=-1)
        k = torch.nn.functional.softmax(k, dim=-2)
        
        # Einstein notation for efficient computation
        context = torch.einsum("bhcxyz,bhdxyz->bhcxyz", k, v)
        out = torch.einsum("bhcxyz,bhdxyz->bhdxyz", q, context)
        
        out = out.reshape(b, self.heads * self.dim_head, d, h, w)
        return self.to_out(out) + x  # Residual connection


class ResnetBlock3D(nn.Module):
    """ResNet block adapted for 3D with memory optimizations."""
    
    def __init__(self, dim, dim_out, time_dim, dropout=0.1, use_checkpoint=True):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        
        # First convolution branch
        self.block1 = nn.Sequential(
            nn.GroupNorm(8, dim),
            nn.SiLU(),
            nn.Conv3d(dim, dim_out, 3, padding=1)
        )
        
        # Time embedding projection
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, dim_out)
        )
        
        # Second convolution branch
        self.block2 = nn.Sequential(
            nn.GroupNorm(8, dim_out),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv3d(dim_out, dim_out, 3, padding=1)
        )
        
        # Residual connection with projection if dimensions change
        self.res_conv = nn.Conv3d(dim, dim_out, 1) if dim != dim_out else nn.Identity()
        
    def forward(self, x, time_emb):
        """Forward pass with optional gradient checkpointing.
        
        Args:
            x: Input feature volume [B, C, D, H, W]
            time_emb: Time embedding [B, time_dim]
            
        Returns:
            Updated features
        """
        if self.use_checkpoint and self.training:
            return torch.utils.checkpoint.checkpoint(self._forward, x, time_emb)
        else:
            return self._forward(x, time_emb)
    
    def _forward(self, x, time_emb):
        # Main branch
        h = self.block1(x)
        
        # Add time embedding
        time_emb = self.time_mlp(time_emb)[:, :, None, None, None]
        h = h + time_emb
        
        # Second convolutional block
        h = self.block2(h)
        
        # Residual connection
        return h + self.res_conv(x)


class UNet3D(nn.Module):
    """Complete 3D UNet architecture with attention and conditioning."""
    
    def __init__(
        self, 
        in_channels=1, 
        out_channels=1, 
        dim=64, 
        dim_mults=(1, 2, 4, 8), 
        num_res_blocks=2,
        attention_levels=[2, 3],  # Which levels to add attention
        dropout=0.1,
        time_dim=256,
        num_classes=None  # For conditioning
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.time_dim = time_dim
        self.num_classes = num_classes
        
        # Initial projection from image space
        self.init_conv = nn.Conv3d(in_channels, dim, kernel_size=3, padding=1)
        
        # Time embedding
        self.time_embedding = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )
        
        # Optional class embedding for conditioning
        if num_classes is not None:
            self.class_embedding = nn.Embedding(num_classes, time_dim)
        
        # Track dimensions for encoder/decoder
        dims = [dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        
        # Encoder modules
        self.downs = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind == len(in_out) - 1
            use_attention = ind in attention_levels
            
            self.downs.append(nn.ModuleList([
                nn.ModuleList([
                    ResnetBlock3D(dim_in, dim_in, time_dim, dropout)
                    for _ in range(num_res_blocks)
                ]),
                EfficientAttention3D(dim_in) if use_attention else nn.Identity(),
                nn.Conv3d(dim_in, dim_out, 4, 2, 1) if not is_last else nn.Conv3d(dim_in, dim_out, 3, 1, 1)
            ]))
        
        # Middle modules
        mid_dim = dims[-1]
        self.mid_block1 = ResnetBlock3D(mid_dim, mid_dim, time_dim, dropout)
        self.mid_attn = EfficientAttention3D(mid_dim)
        self.mid_block2 = ResnetBlock3D(mid_dim, mid_dim, time_dim, dropout)
        
        # Decoder modules
        self.ups = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == len(in_out) - 1
            use_attention = len(in_out) - 1 - ind in attention_levels
            
            self.ups.append(nn.ModuleList([
                nn.ModuleList([
                    ResnetBlock3D(dim_out + dim_in, dim_out, time_dim, dropout)
                    for _ in range(num_res_blocks + 1)
                ]),
                EfficientAttention3D(dim_out) if use_attention else nn.Identity(),
                nn.ConvTranspose3d(dim_out, dim_in, 4, 2, 1) if not is_last else nn.Conv3d(dim_out, dim_in, 3, 1, 1)
            ]))
        
        # Final output projection
        self.final_res_block = ResnetBlock3D(dim, dim, time_dim, dropout)
        self.final_conv = nn.Conv3d(dim, out_channels, 1)
        
        # Log model size
        n_params = sum(p.numel() for p in self.parameters())
        print(f"Model initialized with {n_params/1e6:.2f}M parameters")
        
    def forward(self, x, timestep, conditioning=None):
        """
        Forward pass of 3D UNet.
        
        Args:
            x: Input tensor [B, C, D, H, W]
            timestep: Diffusion timesteps [B]
            conditioning: Optional class conditioning [B]
            
        Returns:
            Tensor with same shape as input
        """
        # Get time embedding
        t_emb = self.time_embedding(timestep)
        
        # Add class conditioning if provided
        if self.num_classes is not None and conditioning is not None:
            c_emb = self.class_embedding(conditioning)
            t_emb = t_emb + c_emb
        
        # Initial convolution
        x = self.init_conv(x)
        h = x
        
        # Store skip connections for decoder
        skips = []
        
        # Encoder
        for blocks, attention, downsample in self.downs:
            for block in blocks:
                h = block(h, t_emb)
            h = attention(h)
            skips.append(h)
            h = downsample(h)
        
        # Middle
        h = self.mid_block1(h, t_emb)
        h = self.mid_attn(h)
        h = self.mid_block2(h, t_emb)
        
        # Decoder with skip connections
        for blocks, attention, upsample in self.ups:
            skip = skips.pop()
            h = torch.cat([h, skip], dim=1)
            
            for block in blocks:
                h = block(h, t_emb)
                
            h = attention(h)
            h = upsample(h)
        
        # Final blocks
        h = self.final_res_block(h, t_emb)
        h = self.final_conv(h)
        
        return h


def test_unet3d():
    """Test UNet3D architecture with a sample input."""
    model = UNet3D(
        in_channels=1,
        out_channels=1,
        dim=32,  # Smaller for testing
        dim_mults=(1, 2, 4),
        num_res_blocks=1
    )
    
    # Create dummy inputs
    dummy_input = torch.randn(2, 1, 32, 32, 32)  # [B, C, D, H, W]
    dummy_timestep = torch.tensor([0, 10])
    
    # Run forward pass
    output = model(dummy_input, dummy_timestep)
    
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    assert output.shape == dummy_input.shape, "Output shape doesn't match input shape"
    print("UNet3D test passed!")
    
    return model


if __name__ == "__main__":
    test_unet3d()

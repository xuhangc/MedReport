import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers
from einops import rearrange


##########################################################################
## Layer Norm
def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight

class LayerNorm(nn.Module):
    def __init__(self, dim):
        super(LayerNorm, self).__init__()
        self.body = BiasFree_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


##########################################################################
## Novel Component 1: Parallel Shuffle Attention (PSA)
class ParallelShuffleAttention(nn.Module):
    """
    Novel shuffle-based attention mechanism without loops.
    Uses parallel tensor operations for efficient spatial mixing.
    """
    def __init__(self, dim, num_heads=8, bias=False):
        super(ParallelShuffleAttention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, 
                                     padding=1, groups=dim * 3, bias=bias)
        
        # Parallel shuffle projections for different spatial patterns
        self.shuffle_proj_h = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.shuffle_proj_w = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.shuffle_proj_d = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        
    def parallel_shuffle_mix(self, x):
        """Parallel shuffle mixing without loops"""
        B, C, H, W = x.shape
        
        # Horizontal shuffle: split and recombine
        x_h = self.shuffle_proj_h(x)
        x_h = rearrange(x_h, 'b c h w -> b c w h')
        
        # Vertical shuffle: split and recombine
        x_w = self.shuffle_proj_w(x)
        
        # Diagonal shuffle: using einops for efficient diagonal mixing
        x_d = self.shuffle_proj_d(x)
        x_d = rearrange(x_d, 'b c (h h2) (w w2) -> b c (h w2) (h2 w)', 
                       h2=2, w2=2) if H % 2 == 0 and W % 2 == 0 else x_d
        
        # Combine all shuffle patterns
        x_shuffled = x_h + rearrange(x_w, 'b c h w -> b c w h') + x_d
        
        return x_shuffled

    def forward(self, x):
        B, C, H, W = x.shape
        
        # Apply shuffle mixing
        x_shuffled = self.parallel_shuffle_mix(x)
        
        # Standard attention with shuffled features
        qkv = self.qkv_dwconv(self.qkv(x_shuffled))
        q, k, v = qkv.chunk(3, dim=1)
        
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        
        out = (attn @ v)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', 
                       head=self.num_heads, h=H, w=W)
        
        out = self.project_out(out)
        return out


##########################################################################
## Novel Component 2: Adaptive Multi-Axis Gating FFN (AMAGFFN)
class AdaptiveMultiAxisGatingFFN(nn.Module):
    """
    Enhanced MAXIM-based FFN with adaptive gating and multi-axis processing.
    Combines spatial gating with channel-wise modulation.
    """
    def __init__(self, dim, ffn_expansion_factor=2.66, bias=False):
        super(AdaptiveMultiAxisGatingFFN, self).__init__()
        
        hidden_features = int(dim * ffn_expansion_factor)
        
        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)
        
        # Multi-axis depthwise convolutions
        self.dwconv_h = nn.Conv2d(hidden_features, hidden_features, 
                                  kernel_size=(3, 1), padding=(1, 0), 
                                  groups=hidden_features, bias=bias)
        self.dwconv_w = nn.Conv2d(hidden_features, hidden_features, 
                                  kernel_size=(1, 3), padding=(0, 1), 
                                  groups=hidden_features, bias=bias)
        self.dwconv_full = nn.Conv2d(hidden_features, hidden_features, 
                                     kernel_size=3, padding=1, 
                                     groups=hidden_features, bias=bias)
        
        # Adaptive gating mechanism
        self.gate_conv = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(hidden_features, hidden_features // 4, 1, bias=bias),
            nn.GELU(),
            nn.Conv2d(hidden_features // 4, hidden_features, 1, bias=bias),
            nn.Sigmoid()
        )
        
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)
        
    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = x.chunk(2, dim=1)
        
        # Multi-axis processing
        x1_h = self.dwconv_h(x1)
        x1_w = self.dwconv_w(x1)
        x1_full = self.dwconv_full(x1)
        
        # Combine multi-axis features
        x1_combined = x1_h + x1_w + x1_full
        
        # Adaptive gating
        gate = self.gate_conv(x1_combined)
        x1_gated = x1_combined * gate
        
        # Gated activation
        x = F.gelu(x1_gated) * x2
        x = self.project_out(x)
        
        return x


##########################################################################
## Transformer Block
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads=8, ffn_expansion_factor=2.66, bias=False):
        super(TransformerBlock, self).__init__()
        
        self.norm1 = LayerNorm(dim)
        self.attn = ParallelShuffleAttention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim)
        self.ffn = AdaptiveMultiAxisGatingFFN(dim, ffn_expansion_factor, bias)
        
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


##########################################################################
## Downsample and Upsample
class Downsample(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Downsample, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=4, stride=2, padding=1),
        )
        
    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Upsample, self).__init__()
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(in_channel, out_channel, kernel_size=2, stride=2),
        )
        
    def forward(self, x):
        return self.deconv(x)


##########################################################################
## Input/Output Projection
class InputProj(nn.Module):
    def __init__(self, in_channel=3, out_channel=32, kernel_size=3):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, 
                     stride=1, padding=kernel_size // 2),
            nn.LeakyReLU(inplace=True)
        )
        
    def forward(self, x):
        return self.proj(x)


class OutputProj(nn.Module):
    def __init__(self, in_channel=32, out_channel=3, kernel_size=3):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, 
                     stride=1, padding=kernel_size // 2),
        )
        
    def forward(self, x):
        return self.proj(x)


##########################################################################
## Medical Prescription Deblurring Transformer (MPDFormer)
'''
Novel Contribution 1: Parallel Shuffle Attention (PSA)
Loop-free implementation using parallel tensor operations
Three parallel shuffle patterns (horizontal, vertical, diagonal)
Efficient spatial mixing without iterative processing
Uses einops for clean tensor manipulation

Novel Contribution 2: Adaptive Multi-Axis Gating FFN (AMAGFFN)
Enhanced MAXIM-style feed-forward network
Multi-axis depthwise convolutions (horizontal, vertical, full)
Adaptive channel-wise gating mechanism
Combines spatial and channel attention
'''
class MPDFormer(nn.Module):
    """
    Medical Prescription Deblurring Transformer with U-Net structure.
    
    Novel Contributions:
    1. Parallel Shuffle Attention (PSA) - Loop-free shuffle mechanism
    2. Adaptive Multi-Axis Gating FFN (AMAGFFN) - Enhanced MAXIM-based FFN
    
    Args:
        inp_channels: Number of input channels (default: 3)
        out_channels: Number of output channels (default: 3)
        dim: Base feature dimension (default: 32)
        num_blocks: Number of transformer blocks at each level (default: [2, 4, 4, 2])
        num_heads: Number of attention heads at each level (default: [1, 2, 4, 8])
        ffn_expansion_factor: FFN expansion ratio (default: 2.66)
        bias: Use bias in convolutions (default: False)
    """
    def __init__(self, 
                 inp_channels=3,
                 out_channels=3, 
                 dim=48,
                 num_blocks=[5, 7, 7, 9], 
                 num_heads=[1, 2, 4, 8],
                 ffn_expansion_factor=2.66,
                 bias=False):
        super(MPDFormer, self).__init__()
        
        self.input_proj = InputProj(inp_channels, dim)
        
        # Encoder
        self.encoder_level1 = nn.Sequential(*[
            TransformerBlock(dim=dim, num_heads=num_heads[0], 
                           ffn_expansion_factor=ffn_expansion_factor, bias=bias) 
            for _ in range(num_blocks[0])])
        self.down1_2 = Downsample(dim, int(dim * 2))
        
        self.encoder_level2 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2), num_heads=num_heads[1], 
                           ffn_expansion_factor=ffn_expansion_factor, bias=bias) 
            for _ in range(num_blocks[1])])
        self.down2_3 = Downsample(int(dim * 2), int(dim * 4))
        
        self.encoder_level3 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 4), num_heads=num_heads[2], 
                           ffn_expansion_factor=ffn_expansion_factor, bias=bias) 
            for _ in range(num_blocks[2])])
        self.down3_4 = Downsample(int(dim * 4), int(dim * 8))
        
        # Bottleneck
        self.latent = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 8), num_heads=num_heads[3], 
                           ffn_expansion_factor=ffn_expansion_factor, bias=bias) 
            for _ in range(num_blocks[3])])
        
        # Decoder
        self.up4_3 = Upsample(int(dim * 8), int(dim * 4))
        self.reduce_chan_level3 = nn.Conv2d(int(dim * 8), int(dim * 4), 
                                            kernel_size=1, bias=bias)
        self.decoder_level3 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 4), num_heads=num_heads[2], 
                           ffn_expansion_factor=ffn_expansion_factor, bias=bias) 
            for _ in range(num_blocks[2])])
        
        self.up3_2 = Upsample(int(dim * 4), int(dim * 2))
        self.reduce_chan_level2 = nn.Conv2d(int(dim * 4), int(dim * 2), 
                                            kernel_size=1, bias=bias)
        self.decoder_level2 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2), num_heads=num_heads[1], 
                           ffn_expansion_factor=ffn_expansion_factor, bias=bias) 
            for _ in range(num_blocks[1])])
        
        self.up2_1 = Upsample(int(dim * 2), int(dim))
        self.reduce_chan_level1 = nn.Conv2d(int(dim * 2), int(dim), 
                                            kernel_size=1, bias=bias)
        self.decoder_level1 = nn.Sequential(*[
            TransformerBlock(dim=int(dim), num_heads=num_heads[0], 
                           ffn_expansion_factor=ffn_expansion_factor, bias=bias) 
            for _ in range(num_blocks[0])])
        
        self.output_proj = OutputProj(dim, out_channels)
        
    def forward(self, inp_img):
        """
        Args:
            inp_img: Input blurry image tensor [B, 3, H, W]
            
        Returns:
            out_img: Deblurred image tensor [B, 3, H, W]
        """
        # Input projection
        inp_enc_level1 = self.input_proj(inp_img)
        
        # Encoder
        out_enc_level1 = self.encoder_level1(inp_enc_level1)
        inp_enc_level2 = self.down1_2(out_enc_level1)
        
        out_enc_level2 = self.encoder_level2(inp_enc_level2)
        inp_enc_level3 = self.down2_3(out_enc_level2)
        
        out_enc_level3 = self.encoder_level3(inp_enc_level3)
        inp_enc_level4 = self.down3_4(out_enc_level3)
        
        # Bottleneck
        latent = self.latent(inp_enc_level4)
        
        # Decoder
        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        out_dec_level3 = self.decoder_level3(inp_dec_level3)
        
        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2)
        
        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        inp_dec_level1 = self.reduce_chan_level1(inp_dec_level1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)
        
        # Output projection
        out_img = self.output_proj(out_dec_level1) + inp_img
        
        return out_img


##########################################################################
## Usage Example
if __name__ == '__main__':
    # Create model
    model = MPDFormer().cuda()
    model.eval()

    # Test with different image sizes
    test_sizes = [(256, 256)]
    
    for h, w in test_sizes:
        x = torch.randn(1, 3, h, w).cuda()
        y = model(x)
        print(f"Input: {x.shape} -> Output: {y.shape}")
        assert y.shape == x.shape, "Output shape mismatch!"
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params / 1e6:.2f}M")

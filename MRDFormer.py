import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import trunc_normal_

##############################################################################
# CONTRIBUTION 1: Adaptive Histogram-Guided Attention (AHGA)
# Combines histogram sorting from Histoformer with channel-spatial attention from CASAB
##############################################################################

class AdaptiveHistogramGuidedAttention(nn.Module):
    """
    Novel contribution: Merges histogram-based feature sorting with 
    dual channel-spatial attention for medical image deblurring.
    """
    def __init__(self, dim, num_heads=8, reduction=16):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        
        # Histogram-guided query, key, value projection
        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=False)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, padding=1, 
                                     groups=dim*3, bias=False)
        
        # Channel Attention (from CASAB)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.channel_fc = nn.Sequential(
            nn.Conv2d(dim, dim // reduction, kernel_size=1, bias=False),
            nn.SiLU(),
            nn.Conv2d(dim // reduction, dim, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
        
        # Spatial Attention (enhanced from CASAB)
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(4, 1, kernel_size=7, padding=3),
            nn.SiLU(),
            nn.Conv2d(1, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.norm = nn.LayerNorm(dim)
        
    def forward(self, x):
        b, c, h, w = x.shape
        
        # Histogram-guided sorting (from Histoformer)
        x_sort, idx_h = x[:, :c//2].sort(-2)
        x_sort, idx_w = x_sort.sort(-1)
        x[:, :c//2] = x_sort
        
        # Generate Q, K, V
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)
        
        # Sort Q, K, V based on histogram
        v_flat, idx = v.view(b, c, -1).sort(dim=-1)
        q_flat = torch.gather(q.view(b, c, -1), dim=2, index=idx)
        k_flat = torch.gather(k.view(b, c, -1), dim=2, index=idx)
        
        # Multi-head attention
        q_flat = q_flat.reshape(b, self.num_heads, c // self.num_heads, -1)
        k_flat = k_flat.reshape(b, self.num_heads, c // self.num_heads, -1)
        v_flat = v_flat.reshape(b, self.num_heads, c // self.num_heads, -1)
        
        q_flat = F.normalize(q_flat, dim=-1)
        k_flat = F.normalize(k_flat, dim=-1)
        
        attn = (q_flat @ k_flat.transpose(-2, -1)) * self.temperature
        attn = F.softmax(attn, dim=-1)
        
        out = (attn @ v_flat).reshape(b, c, -1)
        out = torch.scatter(out, 2, idx, out).view(b, c, h, w)
        
        # Channel Attention
        ca_avg = self.channel_fc(self.avg_pool(out))
        ca_max = self.channel_fc(self.max_pool(out))
        ca = ca_avg + ca_max
        out = out * ca
        
        # Spatial Attention
        mean_out = torch.mean(out, dim=1, keepdim=True)
        max_out, _ = torch.max(out, dim=1, keepdim=True)
        min_out, _ = torch.min(out, dim=1, keepdim=True)
        sum_out = torch.sum(out, dim=1, keepdim=True)
        spatial_pool = torch.cat([mean_out, max_out, min_out, sum_out], dim=1)
        sa = self.spatial_conv(spatial_pool)
        out = out * sa
        
        out = self.project_out(out)
        
        # Reverse histogram sorting
        out_replace = out[:, :c//2]
        out_replace = torch.scatter(out_replace, -1, idx_w, out_replace)
        out_replace = torch.scatter(out_replace, -2, idx_h, out_replace)
        out[:, :c//2] = out_replace
        
        return out


##############################################################################
# CONTRIBUTION 2: Multi-Scale Window-Grid Transformer Block (MSWGT)
# Combines MaxViT's window-grid attention with multi-scale processing
##############################################################################

def window_partition(x, window_size):
    B, C, H, W = x.shape
    x = x.view(B, C, H // window_size, window_size, W // window_size, window_size)
    windows = x.permute(0, 2, 4, 3, 5, 1).contiguous().view(-1, window_size, window_size, C)
    return windows

def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 5, 1, 3, 2, 4).contiguous().view(B, -1, H, W)
    return x

def grid_partition(x, grid_size):
    B, C, H, W = x.shape
    x = x.view(B, C, grid_size, H // grid_size, grid_size, W // grid_size)
    grid = x.permute(0, 3, 5, 2, 4, 1).contiguous().view(-1, grid_size, grid_size, C)
    return grid

def grid_reverse(grid, grid_size, H, W):
    B = int(grid.shape[0] / (H * W / grid_size / grid_size))
    x = grid.view(B, H // grid_size, W // grid_size, grid_size, grid_size, -1)
    x = x.permute(0, 5, 2, 1, 3, 4).contiguous().view(B, -1, H, W)
    return x


class MultiScaleWindowGridTransformer(nn.Module):
    """
    Novel contribution: Multi-scale processing with both window and grid attention
    for capturing local and global blur patterns in medical reports.
    """
    def __init__(self, dim, num_heads=8, window_size=8, mlp_ratio=4., drop=0.):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        
        # Multi-scale feature extraction
        self.scale_conv1 = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.scale_conv2 = nn.Conv2d(dim, dim, kernel_size=5, padding=2, groups=dim)
        self.scale_conv3 = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.scale_fusion = nn.Conv2d(dim*3, dim, kernel_size=1)
        
        # Window attention
        self.norm_window = nn.LayerNorm(dim)
        self.window_attn = nn.MultiheadAttention(dim, num_heads, dropout=drop, batch_first=True)
        
        # Grid attention
        self.norm_grid = nn.LayerNorm(dim)
        self.grid_attn = nn.MultiheadAttention(dim, num_heads, dropout=drop, batch_first=True)
        
        # MLP
        self.norm_mlp = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(drop)
        )
        
    def forward(self, x):
        B, C, H, W = x.shape
        
        # Multi-scale feature extraction
        scale1 = self.scale_conv1(x)
        scale2 = self.scale_conv2(x)
        scale3 = self.scale_conv3(x)
        x_multi = self.scale_fusion(torch.cat([scale1, scale2, scale3], dim=1))
        
        # Window attention
        x_window = window_partition(x_multi, self.window_size)  # (B*num_windows, window_size, window_size, C)
        x_window = x_window.view(-1, self.window_size * self.window_size, C)
        x_window = self.norm_window(x_window)
        x_window_attn, _ = self.window_attn(x_window, x_window, x_window)
        x_window_attn = x_window_attn.view(-1, self.window_size, self.window_size, C)
        x_window_out = window_reverse(x_window_attn, self.window_size, H, W)
        
        x = x + x_window_out
        
        # Grid attention
        x_grid = grid_partition(x, self.window_size)
        x_grid = x_grid.view(-1, self.window_size * self.window_size, C)
        x_grid = self.norm_grid(x_grid)
        x_grid_attn, _ = self.grid_attn(x_grid, x_grid, x_grid)
        x_grid_attn = x_grid_attn.view(-1, self.window_size, self.window_size, C)
        x_grid_out = grid_reverse(x_grid_attn, self.window_size, H, W)
        
        x = x + x_grid_out
        
        # MLP
        x_flat = x.flatten(2).transpose(1, 2)  # B, H*W, C
        x_flat = x_flat + self.mlp(self.norm_mlp(x_flat))
        x = x_flat.transpose(1, 2).view(B, C, H, W)
        
        return x


##############################################################################
# Building Blocks
##############################################################################

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )
        
    def forward(self, x):
        return self.conv(x)


class Downsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.down = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )
        
    def forward(self, x):
        return self.down(x)


class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )
        
    def forward(self, x):
        return self.up(x)


##############################################################################
# Main Model: Medical Report Deblurring Transformer (MRD-Former)
##############################################################################
'''
Two Novel Contributions:
Adaptive Histogram-Guided Attention (AHGA):
1. Merges histogram-based feature sorting from Histoformer
2. Integrates dual channel-spatial attention from CASAB
3. Specifically designed for medical image feature enhancement
4. Handles varying intensity distributions in medical reports
Multi-Scale Window-Grid Transformer (MSWGT):
1. Combines MaxViT's window-grid attention mechanism
2. Multi-scale feature extraction (3×3, 5×5, 7×7 kernels)
3. Captures both local blur (window attention) and global blur patterns (grid attention)
4. Efficient for processing medical documents with varying blur scales
'''
class MRDFormer(nn.Module):
    """
    Medical Report Deblurring Transformer with U-Net structure.
    
    Two Novel Contributions:
    1. Adaptive Histogram-Guided Attention (AHGA): Combines histogram sorting 
       with dual channel-spatial attention for medical image feature enhancement.
    2. Multi-Scale Window-Grid Transformer (MSWGT): Multi-scale processing with 
       both window and grid attention for capturing local and global blur patterns.
    
    Args:
        in_channels: Number of input channels (default: 3 for RGB, 1 for grayscale)
        out_channels: Number of output channels (default: 3)
        embed_dim: Base embedding dimension (default: 64)
        depths: Depth of each stage [encoder1, encoder2, encoder3, encoder4, bottleneck, 
                                     decoder1, decoder2, decoder3, decoder4]
        num_heads: Number of attention heads at each stage
        window_size: Window size for window-grid attention (default: 7)
        mlp_ratio: MLP expansion ratio (default: 4.0)
        drop_rate: Dropout rate (default: 0.1)
    """
    def __init__(
        self,
        in_channels=3,
        out_channels=3,
        embed_dim=64,
        depths=[2, 2, 4, 2, 4, 2, 4, 2, 2],
        num_heads=[2, 4, 8, 16, 16, 16, 8, 4, 2],
        window_size=8,
        mlp_ratio=4.,
        drop_rate=0.1
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.embed_dim = embed_dim
        
        # Input projection
        self.input_proj = nn.Sequential(
            nn.Conv2d(in_channels, embed_dim, 3, padding=1),
            nn.BatchNorm2d(embed_dim),
            nn.GELU()
        )
        
        # Encoder Stage 1
        self.enc1_conv = ConvBlock(embed_dim, embed_dim)
        self.enc1_ahga = nn.ModuleList([
            AdaptiveHistogramGuidedAttention(embed_dim, num_heads[0]) 
            for _ in range(depths[0])
        ])
        self.down1 = Downsample(embed_dim, embed_dim*2)
        
        # Encoder Stage 2
        self.enc2_conv = ConvBlock(embed_dim*2, embed_dim*2)
        self.enc2_mswgt = nn.ModuleList([
            MultiScaleWindowGridTransformer(embed_dim*2, num_heads[1], window_size, mlp_ratio, drop_rate)
            for _ in range(depths[1])
        ])
        self.down2 = Downsample(embed_dim*2, embed_dim*4)
        
        # Encoder Stage 3
        self.enc3_conv = ConvBlock(embed_dim*4, embed_dim*4)
        self.enc3_ahga = nn.ModuleList([
            AdaptiveHistogramGuidedAttention(embed_dim*4, num_heads[2])
            for _ in range(depths[2])
        ])
        self.down3 = Downsample(embed_dim*4, embed_dim*8)
        
        # Encoder Stage 4
        self.enc4_conv = ConvBlock(embed_dim*8, embed_dim*8)
        self.enc4_mswgt = nn.ModuleList([
            MultiScaleWindowGridTransformer(embed_dim*8, num_heads[3], window_size, mlp_ratio, drop_rate)
            for _ in range(depths[3])
        ])
        self.down4 = Downsample(embed_dim*8, embed_dim*16)
        
        # Bottleneck
        self.bottleneck_conv = ConvBlock(embed_dim*16, embed_dim*16)
        self.bottleneck_blocks = nn.ModuleList([
            nn.Sequential(
                AdaptiveHistogramGuidedAttention(embed_dim*16, num_heads[4]),
                MultiScaleWindowGridTransformer(embed_dim*16, num_heads[4], window_size, mlp_ratio, drop_rate)
            )
            for _ in range(depths[4])
        ])
        
        # Decoder Stage 1
        self.up1 = Upsample(embed_dim*16, embed_dim*8)
        self.dec1_conv = ConvBlock(embed_dim*16, embed_dim*8)
        self.dec1_mswgt = nn.ModuleList([
            MultiScaleWindowGridTransformer(embed_dim*8, num_heads[5], window_size, mlp_ratio, drop_rate)
            for _ in range(depths[5])
        ])
        
        # Decoder Stage 2
        self.up2 = Upsample(embed_dim*8, embed_dim*4)
        self.dec2_conv = ConvBlock(embed_dim*8, embed_dim*4)
        self.dec2_ahga = nn.ModuleList([
            AdaptiveHistogramGuidedAttention(embed_dim*4, num_heads[6])
            for _ in range(depths[6])
        ])
        
        # Decoder Stage 3
        self.up3 = Upsample(embed_dim*4, embed_dim*2)
        self.dec3_conv = ConvBlock(embed_dim*4, embed_dim*2)
        self.dec3_mswgt = nn.ModuleList([
            MultiScaleWindowGridTransformer(embed_dim*2, num_heads[7], window_size, mlp_ratio, drop_rate)
            for _ in range(depths[7])
        ])
        
        # Decoder Stage 4
        self.up4 = Upsample(embed_dim*2, embed_dim)
        self.dec4_conv = ConvBlock(embed_dim*2, embed_dim)
        self.dec4_ahga = nn.ModuleList([
            AdaptiveHistogramGuidedAttention(embed_dim, num_heads[8])
            for _ in range(depths[8])
        ])
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, 3, padding=1),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
            nn.Conv2d(embed_dim, out_channels, 1)
        )
        
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Forward pass
        Args:
            x: Input image tensor of shape (B, C, H, W)
        Returns:
            Deblurred image tensor of shape (B, C, H, W)
        """
        # Input projection
        x = self.input_proj(x)
        
        # Encoder Stage 1
        enc1 = self.enc1_conv(x)
        for ahga in self.enc1_ahga:
            enc1 = enc1 + ahga(enc1)
        
        # Encoder Stage 2
        x = self.down1(enc1)
        enc2 = self.enc2_conv(x)
        for mswgt in self.enc2_mswgt:
            enc2 = mswgt(enc2)
        
        # Encoder Stage 3
        x = self.down2(enc2)
        enc3 = self.enc3_conv(x)
        for ahga in self.enc3_ahga:
            enc3 = enc3 + ahga(enc3)
        
        # Encoder Stage 4
        x = self.down3(enc3)
        enc4 = self.enc4_conv(x)
        for mswgt in self.enc4_mswgt:
            enc4 = mswgt(enc4)
        
        # Bottleneck
        x = self.down4(enc4)
        x = self.bottleneck_conv(x)
        for block in self.bottleneck_blocks:
            ahga, mswgt = block
            x = x + ahga(x)
            x = mswgt(x)
        
        # Decoder Stage 1
        x = self.up1(x)
        x = torch.cat([x, enc4], dim=1)
        x = self.dec1_conv(x)
        for mswgt in self.dec1_mswgt:
            x = mswgt(x)
        
        # Decoder Stage 2
        x = self.up2(x)
        x = torch.cat([x, enc3], dim=1)
        x = self.dec2_conv(x)
        for ahga in self.dec2_ahga:
            x = x + ahga(x)
        
        # Decoder Stage 3
        x = self.up3(x)
        x = torch.cat([x, enc2], dim=1)
        x = self.dec3_conv(x)
        for mswgt in self.dec3_mswgt:
            x = mswgt(x)
        
        # Decoder Stage 4
        x = self.up4(x)
        x = torch.cat([x, enc1], dim=1)
        x = self.dec4_conv(x)
        for ahga in self.dec4_ahga:
            x = x + ahga(x)
        
        # Output projection
        x = self.output_proj(x)
        
        return x


##############################################################################
# Model Variants
##############################################################################
def mrdformer_tiny(in_channels=3, out_channels=3):
    """Tiny variant for fast inference"""
    return MRDFormer(
        in_channels=in_channels,
        out_channels=out_channels,
        embed_dim=32,
        depths=[1, 1, 2, 1, 2, 1, 2, 1, 1],
        num_heads=[2, 4, 8, 16, 16, 16, 8, 4, 2],
        window_size=8
    )

def mrdformer_small(in_channels=3, out_channels=3):
    """Small variant for balanced performance"""
    return MRDFormer(
        in_channels=in_channels,
        out_channels=out_channels,
        embed_dim=48,
        depths=[2, 2, 3, 2, 3, 2, 3, 2, 2],
        num_heads=[2, 4, 8, 16, 16, 16, 8, 4, 2],
        window_size=8
    )

def mrdformer_base(in_channels=3, out_channels=3):
    """Base variant (default)"""
    return MRDFormer(
        in_channels=in_channels,
        out_channels=out_channels,
        embed_dim=64,
        depths=[2, 2, 4, 2, 4, 2, 4, 2, 2],
        num_heads=[2, 4, 8, 16, 16, 16, 8, 4, 2],
        window_size=8
    )

def mrdformer_large(in_channels=3, out_channels=3):
    """Large variant for best quality"""
    return MRDFormer(
        in_channels=in_channels,
        out_channels=out_channels,
        embed_dim=96,
        depths=[3, 3, 6, 3, 6, 3, 6, 3, 3],
        num_heads=[3, 6, 12, 24, 24, 24, 12, 6, 3],
        window_size=8
    )


##############################################################################
# Testing and Usage Example
##############################################################################

if __name__ == '__main__':
    # Test the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model variants
    models = {
        'tiny': mrdformer_tiny(in_channels=3, out_channels=3),
        'small': mrdformer_small(in_channels=3, out_channels=3),
        'base': mrdformer_base(in_channels=3, out_channels=3),
        'large': mrdformer_large(in_channels=3, out_channels=3)
    }
    
    # Test input
    x = torch.randn(1, 3, 256, 256).to(device)
    
    print("=" * 80)
    print("Medical Report Deblurring Transformer (MRD-Former)")
    print("=" * 80)
    print("\nTwo Novel Contributions:")
    print("1. Adaptive Histogram-Guided Attention (AHGA)")
    print("   - Combines histogram sorting with dual channel-spatial attention")
    print("   - Enhances feature representation for medical images")
    print("\n2. Multi-Scale Window-Grid Transformer (MSWGT)")
    print("   - Multi-scale processing with window and grid attention")
    print("   - Captures both local and global blur patterns")
    print("=" * 80)
    
    for name, model in models.items():
        model = model.to(device)
        model.eval()
        
        with torch.no_grad():
            output = model(x)
        
        # Count parameters
        num_params = sum(p.numel() for p in model.parameters())
        num_params_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"\n{name.upper()} Model:")
        print(f"  Input shape:  {x.shape}")
        print(f"  Output shape: {output.shape}")
        print(f"  Parameters:   {num_params:,} ({num_params_trainable:,} trainable)")
        print(f"  Model size:   {num_params * 4 / 1024 / 1024:.2f} MB (FP32)")
    
    print("\n" + "=" * 80)
    print("Usage Example:")
    print("=" * 80)
    print("""
# For grayscale medical reports
model = mrdformer_base(in_channels=1, out_channels=1)

# For RGB medical images
model = mrdformer_base(in_channels=3, out_channels=3)

# Training
model.train()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
criterion = nn.L1Loss()

for blurry_img, sharp_img in dataloader:
    output = model(blurry_img)
    loss = criterion(output, sharp_img)
    loss.backward()
    optimizer.step()

# Inference
model.eval()
with torch.no_grad():
    deblurred = model(blurry_image)
    """)
    print("=" * 80)

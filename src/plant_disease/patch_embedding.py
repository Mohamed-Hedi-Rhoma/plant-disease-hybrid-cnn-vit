import torch
import torch.nn as nn
import math

class PatchEmbedding(nn.Module):
    """
    Patch Embedding Layer for Hybrid CNN + ViT
    
    Purpose: Convert CNN feature maps into sequence of patches for ViT
    Flow: [batch, 256, 48, 48] → [batch, 36, 512] (36 patches of 512-dim embeddings)
    
    This layer acts as a bridge between CNN and ViT:
    1. Takes CNN feature maps as input (not raw images)
    2. Splits feature maps into non-overlapping patches
    3. Flattens and projects patches to embedding dimension
    4. Adds learnable position embeddings
    """
    
    def __init__(self, feature_size=48, patch_size=8, in_channels=256, embed_dim=512):
        """
        Initialize Patch Embedding layer
        
        Args:
            feature_size (int): Size of input feature maps (48 for 48x48)
            patch_size (int): Size of each patch (8 for 8x8 patches)
            in_channels (int): Number of input channels from CNN (256)
            embed_dim (int): Embedding dimension for ViT (512)
        """
        super(PatchEmbedding, self).__init__()
        
        # Store configuration
        self.feature_size = feature_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        
        # Calculate number of patches
        self.num_patches_per_dim = feature_size // patch_size  # 48 // 8 = 6
        self.num_patches = self.num_patches_per_dim ** 2       # 6 * 6 = 36
        
        # Each patch contains: patch_size * patch_size * in_channels values
        self.patch_dim = patch_size * patch_size * in_channels  # 8 * 8 * 256 = 16,384
        
        # Patch extraction using convolution
        # This is equivalent to splitting into patches and flattening
        self.patch_conv = nn.Conv2d(
            in_channels=in_channels,     # 256 input channels
            out_channels=embed_dim,      # 512 output channels (embedding dim)
            kernel_size=patch_size,      # 8x8 kernel
            stride=patch_size,           # 8x8 stride (no overlap)
            padding=0,                   # No padding
            bias=True                    # Include bias
        )
        
        # Alternative: Linear projection (more explicit)
        # self.patch_projection = nn.Linear(self.patch_dim, embed_dim)
        
        # Learnable position embeddings
        # Each of the 36 patches gets a unique position embedding
        self.position_embeddings = nn.Parameter(
            torch.randn(1, self.num_patches, embed_dim)
        )
        
        # Dropout for regularization
        self.dropout = nn.Dropout(p=0.1)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """
        Initialize weights using appropriate strategies
        """
        # Initialize patch convolution
        nn.init.kaiming_normal_(self.patch_conv.weight, mode='fan_out', nonlinearity='linear')
        if self.patch_conv.bias is not None:
            nn.init.constant_(self.patch_conv.bias, 0)
        
        # Initialize position embeddings with small random values
        nn.init.trunc_normal_(self.position_embeddings, std=0.02)
    
    def forward(self, x):
        """
        Forward pass through patch embedding
        
        Args:
            x (torch.Tensor): CNN feature maps [batch_size, 256, 48, 48]
            
        Returns:
            torch.Tensor: Patch embeddings [batch_size, 36, 512]
        """
        # Input validation
        assert x.dim() == 4, f"Expected 4D input (batch, channels, height, width), got {x.dim()}D"
        assert x.size(1) == self.in_channels, f"Expected {self.in_channels} channels, got {x.size(1)}"
        assert x.size(2) == self.feature_size, f"Expected height {self.feature_size}, got {x.size(2)}"
        assert x.size(3) == self.feature_size, f"Expected width {self.feature_size}, got {x.size(3)}"
        
        batch_size = x.size(0)
        
        # Step 1: Extract patches using convolution
        # Input: [batch, 256, 48, 48]
        # Output: [batch, 512, 6, 6] (512 = embed_dim, 6x6 = num_patches_per_dim)
        patches = self.patch_conv(x)
        
        # Step 2: Reshape to sequence format
        # [batch, 512, 6, 6] → [batch, 512, 36] → [batch, 36, 512]
        patches = patches.flatten(2)          # [batch, 512, 36]
        patches = patches.transpose(1, 2)     # [batch, 36, 512]
        
        # Step 3: Add position embeddings
        # Each patch gets a unique position embedding
        # position_embeddings: [1, 36, 512]
        # patches: [batch, 36, 512]
        patches = patches + self.position_embeddings
        
        # Step 4: Apply dropout
        patches = self.dropout(patches)
        
        return patches
    
    def get_patch_info(self):
        """
        Get information about patch configuration
        
        Returns:
            dict: Patch configuration details
        """
        return {
            'num_patches': self.num_patches,
            'patch_size': self.patch_size,
            'patches_per_dim': self.num_patches_per_dim,
            'patch_dim': self.patch_dim,
            'embed_dim': self.embed_dim
        }
    
    def visualize_patches(self, x):
        """
        Visualize how patches are extracted (for debugging)
        
        Args:
            x (torch.Tensor): Input feature maps [batch, 256, 48, 48]
            
        Returns:
            torch.Tensor: Reshaped patches [batch, 36, 8, 8, 256]
        """
        batch_size = x.size(0)
        
        # Unfold operation to extract patches
        patches = x.unfold(2, self.patch_size, self.patch_size)  # [batch, 256, 6, 48, 8]
        patches = patches.unfold(3, self.patch_size, self.patch_size)  # [batch, 256, 6, 6, 8, 8]
        
        # Rearrange to [batch, num_patches, patch_size, patch_size, channels]
        patches = patches.permute(0, 2, 3, 4, 5, 1)  # [batch, 6, 6, 8, 8, 256]
        patches = patches.reshape(batch_size, self.num_patches, self.patch_size, self.patch_size, self.in_channels)
        
        return patches
    
    def print_model_info(self):
        """
        Print model architecture information
        """
        print("🔗 Patch Embedding Architecture:")
        print("=" * 50)
        print(f"Input:  [batch, {self.in_channels}, {self.feature_size}, {self.feature_size}]")
        print(f"Patches: {self.num_patches} patches of {self.patch_size}x{self.patch_size}")
        print(f"Patch dim: {self.patch_dim} → Embed dim: {self.embed_dim}")
        print(f"Output: [batch, {self.num_patches}, {self.embed_dim}]")
        print("=" * 50)
        
        # Count parameters
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Position embeddings: {self.position_embeddings.numel():,}")
        print(f"Patch projection: {self.patch_conv.weight.numel() + self.patch_conv.bias.numel():,}")


class CombinedCNNPatchEmbedding(nn.Module):
    """
    Combined CNN + Patch Embedding for testing
    Combines the CNN backbone with patch embedding
    """
    
    def __init__(self, cnn_backbone, patch_embedding):
        """
        Initialize combined model
        
        Args:
            cnn_backbone: CNN backbone model
            patch_embedding: Patch embedding layer
        """
        super(CombinedCNNPatchEmbedding, self).__init__()
        self.cnn_backbone = cnn_backbone
        self.patch_embedding = patch_embedding
    
    def forward(self, x):
        """
        Forward pass through CNN + Patch Embedding
        
        Args:
            x (torch.Tensor): Raw images [batch, 3, 384, 384]
            
        Returns:
            torch.Tensor: Patch embeddings [batch, 36, 512]
        """
        # Extract CNN features
        cnn_features = self.cnn_backbone(x)  # [batch, 256, 48, 48]
        
        # Convert to patches
        patch_embeddings = self.patch_embedding(cnn_features)  # [batch, 36, 512]
        
        return patch_embeddings


# Example usage and testing
def test_patch_embedding():
    """
    Test the Patch Embedding layer
    """
    print("🧪 Testing Patch Embedding...")
    
    # Create patch embedding
    patch_embed = PatchEmbedding(
        feature_size=48,    # From CNN output
        patch_size=8,       # 8x8 patches
        in_channels=256,    # From CNN output
        embed_dim=512       # ViT embedding dimension
    )
    
    patch_embed.print_model_info()
    
    # Create sample CNN features (simulating CNN backbone output)
    batch_size = 32
    sample_features = torch.randn(batch_size, 256, 48, 48)
    
    print(f"\n📊 Testing with sample CNN features:")
    print(f"Input shape: {sample_features.shape}")
    
    # Forward pass
    patch_embed.eval()
    with torch.no_grad():
        output = patch_embed(sample_features)
    
    print(f"Output shape: {output.shape}")
    print(f"Expected: [32, 36, 512]")
    
    # Verify output
    expected_shape = (batch_size, 36, 512)
    assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"
    
    print("✅ Patch Embedding test passed!")
    
    # Test patch info
    info = patch_embed.get_patch_info()
    print(f"\n📋 Patch Info:")
    for key, value in info.items():
        print(f"  {key}: {value}")


def test_combined_cnn_patch():
    """
    Test CNN + Patch Embedding combination
    """
    print("\n🔗 Testing Combined CNN + Patch Embedding...")
    
    # Import CNN backbone (assuming it's in the same directory)
    try:
        from plant_disease.cnn_backbone import CNNBackbone
        
        # Create components
        cnn = CNNBackbone(in_channels=3, out_channels=256)
        patch_embed = PatchEmbedding(
            feature_size=48,
            patch_size=8,
            in_channels=256,
            embed_dim=512
        )
        
        # Combine them
        combined_model = CombinedCNNPatchEmbedding(cnn, patch_embed)
        
        # Test with raw images (like your DataLoader)
        batch_size = 32
        raw_images = torch.randn(batch_size, 3, 384, 384)
        
        print(f"📊 Testing end-to-end:")
        print(f"Input (raw images): {raw_images.shape}")
        
        # Forward pass
        combined_model.eval()
        with torch.no_grad():
            patch_embeddings = combined_model(raw_images)
        
        print(f"Output (patch embeddings): {patch_embeddings.shape}")
        print(f"Expected: [32, 36, 512]")
        
        # Verify
        expected_shape = (batch_size, 36, 512)
        assert patch_embeddings.shape == expected_shape, f"Expected {expected_shape}, got {patch_embeddings.shape}"
        
        print("✅ Combined CNN + Patch Embedding test passed!")
        print("🎯 Ready for ViT Transformer!")
        
    except ImportError:
        print("❌ Could not import CNNBackbone. Run cnn_backbone.py first!")


if __name__ == "__main__":
    test_patch_embedding()
    test_combined_cnn_patch()
import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNBackbone(nn.Module):
    """
    CNN Backbone for Hybrid CNN + ViT Architecture
    
    Purpose: Extract rich feature representations from raw images
    Flow: 384x384x3 → 48x48x256 (8x spatial reduction, 256 feature channels)
    
    This CNN acts as a feature extractor that:
    1. Captures local patterns (edges, textures, disease symptoms)
    2. Reduces spatial dimensions for efficient ViT processing
    3. Increases feature depth for richer representations
    """
    
    def __init__(self, in_channels=3, out_channels=256):
        """
        Initialize CNN backbone
        
        Args:
            in_channels (int): Input channels (3 for RGB images)
            out_channels (int): Output feature channels (256 for rich features)
        """
        super(CNNBackbone, self).__init__()
        
        # Store configuration
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Layer 1: Initial feature extraction
        # 384x384x3 → 192x192x64
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,     # 3 (RGB)
            out_channels=64,             # 64 feature maps
            kernel_size=3,               # 3x3 kernel
            stride=1,                    # No striding
            padding=1,                   # Same padding
            bias=False                   # No bias (BatchNorm handles it)
        )
        self.bn1 = nn.BatchNorm2d(64)    # Batch normalization for stability
        self.pool1 = nn.MaxPool2d(
            kernel_size=2,               # 2x2 pooling
            stride=2                     # 2x spatial reduction
        )
        
        # Layer 2: Intermediate feature extraction
        # 192x192x64 → 96x96x128
        self.conv2 = nn.Conv2d(
            in_channels=64,              # From previous layer
            out_channels=128,            # 128 feature maps
            kernel_size=3,               # 3x3 kernel
            stride=1,                    # No striding
            padding=1,                   # Same padding
            bias=False                   # No bias
        )
        self.bn2 = nn.BatchNorm2d(128)   # Batch normalization
        self.pool2 = nn.MaxPool2d(
            kernel_size=2,               # 2x2 pooling
            stride=2                     # 2x spatial reduction
        )
        
        # Layer 3: High-level feature extraction
        # 96x96x128 → 48x48x256
        self.conv3 = nn.Conv2d(
            in_channels=128,             # From previous layer
            out_channels=out_channels,   # 256 feature maps (final)
            kernel_size=3,               # 3x3 kernel
            stride=1,                    # No striding
            padding=1,                   # Same padding
            bias=False                   # No bias
        )
        self.bn3 = nn.BatchNorm2d(out_channels)  # Batch normalization
        self.pool3 = nn.MaxPool2d(
            kernel_size=2,               # 2x2 pooling
            stride=2                     # 2x spatial reduction
        )
        
        # Dropout for regularization
        self.dropout = nn.Dropout2d(p=0.1)  # 10% dropout on feature maps
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """
        Initialize network weights using He initialization
        Good for ReLU activations
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # He initialization for conv layers
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                # BatchNorm initialization
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Forward pass through CNN backbone
        
        Args:
            x (torch.Tensor): Input images [batch_size, 3, 384, 384]
            
        Returns:
            torch.Tensor: Feature maps [batch_size, 256, 48, 48]
        """
        # Input validation
        assert x.dim() == 4, f"Expected 4D input (batch, channels, height, width), got {x.dim()}D"
        assert x.size(1) == self.in_channels, f"Expected {self.in_channels} channels, got {x.size(1)}"
        
        # Store input size for debugging
        batch_size = x.size(0)
        
        # Layer 1: 384x384x3 → 192x192x64
        x = self.conv1(x)                    # [batch, 64, 384, 384]
        x = self.bn1(x)                      # Normalize
        x = F.relu(x, inplace=True)          # ReLU activation
        x = self.pool1(x)                    # [batch, 64, 192, 192]
        
        # Layer 2: 192x192x64 → 96x96x128
        x = self.conv2(x)                    # [batch, 128, 192, 192]
        x = self.bn2(x)                      # Normalize
        x = F.relu(x, inplace=True)          # ReLU activation
        x = self.pool2(x)                    # [batch, 128, 96, 96]
        
        # Layer 3: 96x96x128 → 48x48x256
        x = self.conv3(x)                    # [batch, 256, 96, 96]
        x = self.bn3(x)                      # Normalize
        x = F.relu(x, inplace=True)          # ReLU activation
        x = self.pool3(x)                    # [batch, 256, 48, 48]
        
        # Apply dropout during training
        x = self.dropout(x)                  # [batch, 256, 48, 48]
        
        return x
    
    def get_feature_map_size(self, input_size=384):
        """
        Calculate output feature map size for given input size
        
        Args:
            input_size (int): Input image size (assumed square)
            
        Returns:
            int: Output feature map size
        """
        # Each conv layer maintains size (padding=1, stride=1)
        # Each pool layer reduces by factor of 2
        # Total reduction: 2 * 2 * 2 = 8x
        return input_size // 8
    
    def get_output_channels(self):
        """
        Get number of output feature channels
        
        Returns:
            int: Number of output channels
        """
        return self.out_channels
    
    def print_model_info(self):
        """
        Print model architecture information
        """
        print("🏗️  CNN Backbone Architecture:")
        print("=" * 50)
        print(f"Input:  [batch, 3, 384, 384]")
        print(f"Conv1:  [batch, 64, 384, 384] → Pool → [batch, 64, 192, 192]")
        print(f"Conv2:  [batch, 128, 192, 192] → Pool → [batch, 128, 96, 96]")
        print(f"Conv3:  [batch, 256, 96, 96] → Pool → [batch, 256, 48, 48]")
        print(f"Output: [batch, {self.out_channels}, 48, 48]")
        print("=" * 50)
        
        # Count parameters
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Model size: {total_params * 4 / 1024 / 1024:.2f} MB")  # Assuming float32


# Example usage and testing
def test_cnn_backbone():
    """
    Test the CNN backbone with sample data
    """
    print("🧪 Testing CNN Backbone...")
    
    # Create model
    model = CNNBackbone(in_channels=3, out_channels=256)
    model.print_model_info()
    
    # Create sample input (like your DataLoader output)
    batch_size = 32
    sample_input = torch.randn(batch_size, 3, 384, 384)
    
    print(f"\n📊 Testing with sample input:")
    print(f"Input shape: {sample_input.shape}")
    
    # Forward pass
    model.eval()  # Set to evaluation mode
    with torch.no_grad():
        output = model(sample_input)
    
    print(f"Output shape: {output.shape}")
    print(f"Expected: [32, 256, 48, 48]")
    
    # Verify output
    expected_shape = (batch_size, 256, 48, 48)
    assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"
    
    print("✅ CNN Backbone test passed!")
    
    # Test with different batch sizes
    print("\n🔄 Testing with different batch sizes:")
    for bs in [1, 16, 64]:
        test_input = torch.randn(bs, 3, 384, 384)
        output = model(test_input)
        expected = (bs, 256, 48, 48)
        assert output.shape == expected, f"Batch size {bs}: Expected {expected}, got {output.shape}"
        print(f"✅ Batch size {bs}: {output.shape}")
    
    print("\n🎯 CNN Backbone is ready for integration with ViT!")


if __name__ == "__main__":
    test_cnn_backbone()
import json
import os
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any
import torch

@dataclass
class DataConfig:
    """Configuration for data preprocessing pipeline"""
    
    # Data paths
    data_dir: str = "/home/mrhouma/Documents/Plant_diseases_project/ai_training_data"
    
    # Image preprocessing
    target_size: int = 384
    min_samples_per_class: int = 1500
    clean_augmented: bool = True
    
    # Data splitting
    test_size: float = 0.15
    val_size: float = 0.176  # 0.176 * 0.85 ≈ 0.15 of total
    random_seed: int = 42
    
    # DataLoader settings
    batch_size: int = 32
    num_workers: int = 4
    pin_memory: bool = True
    use_weighted_sampling: bool = True
    
    # Augmentation settings
    augment_rotation_degrees: float = 15.0
    augment_brightness: float = 0.2
    augment_contrast: float = 0.2
    augment_saturation: float = 0.2
    augment_translate: tuple = (0.1, 0.1)


@dataclass
class CNNBackboneConfig:
    """Configuration for CNN backbone"""
    
    # Input/Output channels
    in_channels: int = 3        # RGB images
    out_channels: int = 256     # Feature channels for ViT
    
    # Architecture settings
    conv1_out_channels: int = 64
    conv2_out_channels: int = 128
    conv3_out_channels: int = 256  # Should match out_channels
    
    # Convolution settings
    kernel_size: int = 3
    stride: int = 1
    padding: int = 1
    bias: bool = False  # False because we use BatchNorm
    
    # Pooling settings
    pool_kernel_size: int = 2
    pool_stride: int = 2
    
    # Regularization
    dropout_rate: float = 0.1
    
    # Initialization
    init_mode: str = "kaiming_normal"  # For ReLU
    init_nonlinearity: str = "relu"


@dataclass
class PatchEmbeddingConfig:
    """Configuration for patch embedding layer"""
    
    # Input dimensions (from CNN backbone)
    feature_size: int = 48      # CNN output: 48x48
    in_channels: int = 256      # CNN output channels
    
    # Patch settings
    patch_size: int = 8         # 8x8 patches
    
    # Output dimensions
    embed_dim: int = 512        # ViT embedding dimension
    
    # Regularization
    dropout_rate: float = 0.1
    
    # Position embeddings
    use_position_embeddings: bool = True
    position_embed_init_std: float = 0.02
    
    # Initialization
    init_mode: str = "kaiming_normal"
    conv_bias: bool = True


@dataclass
class AttentionConfig:
    """Configuration for multi-head self-attention"""
    
    # Core attention settings
    embed_dim: int = 512
    num_heads: int = 8
    
    # Regularization
    dropout_rate: float = 0.1
    attention_dropout_rate: float = 0.1
    output_dropout_rate: float = 0.1
    
    # Projections
    qkv_bias: bool = True
    output_bias: bool = True
    
    # Initialization
    init_mode: str = "xavier_uniform"  # Good for attention


@dataclass
class MLPConfig:
    """Configuration for feed-forward MLP"""
    
    # Dimensions
    embed_dim: int = 512
    mlp_ratio: float = 4.0      # hidden_dim = embed_dim * mlp_ratio
    
    # Activation
    activation: str = "gelu"    # gelu, relu, swish
    
    # Regularization
    dropout_rate: float = 0.1
    
    # Initialization
    init_mode: str = "xavier_uniform"
    bias: bool = True


@dataclass
class TransformerBlockConfig:
    """Configuration for transformer block"""
    
    # Core dimensions
    embed_dim: int = 512
    num_heads: int = 8
    mlp_ratio: float = 4.0
    
    # Regularization
    dropout_rate: float = 0.1
    droppath_rate: float = 0.0  # Stochastic depth
    
    # Layer normalization
    layer_norm_eps: float = 1e-6
    pre_norm: bool = True       # Pre-LN vs Post-LN
    
    # Sub-configs
    attention_config: Optional[AttentionConfig] = None
    mlp_config: Optional[MLPConfig] = None
    
    def __post_init__(self):
        """Initialize sub-configs if not provided"""
        if self.attention_config is None:
            self.attention_config = AttentionConfig(
                embed_dim=self.embed_dim,
                num_heads=self.num_heads,
                dropout_rate=self.dropout_rate
            )
        
        if self.mlp_config is None:
            self.mlp_config = MLPConfig(
                embed_dim=self.embed_dim,
                mlp_ratio=self.mlp_ratio,
                dropout_rate=self.dropout_rate
            )


@dataclass
class CLSTokenConfig:
    """Configuration for classification token"""
    
    # Dimensions
    embed_dim: int = 512
    
    # Initialization
    init_std: float = 0.02
    init_mode: str = "trunc_normal"


@dataclass
class ViTConfig:
    """Configuration for complete Vision Transformer"""
    
    # Core architecture
    embed_dim: int = 512
    num_layers: int = 6         # Number of transformer blocks
    num_heads: int = 8
    mlp_ratio: float = 4.0
    
    # Classification
    num_classes: int = 15       # Will be set based on dataset
    
    # Regularization
    dropout_rate: float = 0.1
    droppath_rate: float = 0.1  # Stochastic depth
    
    # Classification head
    classifier_dropout_rate: float = 0.5
    use_classifier_dropout: bool = True
    
    # Sub-configs
    transformer_config: Optional[TransformerBlockConfig] = None
    cls_token_config: Optional[CLSTokenConfig] = None
    
    def __post_init__(self):
        """Initialize sub-configs if not provided"""
        if self.transformer_config is None:
            self.transformer_config = TransformerBlockConfig(
                embed_dim=self.embed_dim,
                num_heads=self.num_heads,
                mlp_ratio=self.mlp_ratio,
                dropout_rate=self.dropout_rate,
                droppath_rate=self.droppath_rate
            )
        
        if self.cls_token_config is None:
            self.cls_token_config = CLSTokenConfig(
                embed_dim=self.embed_dim
            )


@dataclass
class HybridModelConfig:
    """Complete configuration for Hybrid CNN-ViT model"""
    
    # Model name and version
    model_name: str = "HybridCNNViT"
    model_version: str = "v1.0"
    
    # Task settings
    task: str = "plant_disease_classification"
    num_classes: int = 15  # Will be updated based on dataset
    
    # Sub-configurations
    data_config: DataConfig = None
    cnn_config: CNNBackboneConfig = None
    patch_config: PatchEmbeddingConfig = None
    vit_config: ViTConfig = None
    
    # Training settings
    device: str = "auto"  # auto, cpu, cuda
    mixed_precision: bool = True
    compile_model: bool = False  # torch.compile for speed
    
    # Model saving/loading
    save_dir: str = "models"
    checkpoint_format: str = "pytorch"  # pytorch, safetensors
    
    def __post_init__(self):
        """Initialize sub-configs with defaults if not provided"""
        if self.data_config is None:
            self.data_config = DataConfig()
        
        if self.cnn_config is None:
            self.cnn_config = CNNBackboneConfig()
        
        if self.patch_config is None:
            self.patch_config = PatchEmbeddingConfig()
        
        if self.vit_config is None:
            self.vit_config = ViTConfig(num_classes=self.num_classes)
        
        # Ensure consistency
        self._validate_config()
    
    def _validate_config(self):
        """Validate configuration consistency"""
        # CNN output should match patch embedding input
        assert self.cnn_config.out_channels == self.patch_config.in_channels, \
            f"CNN output channels ({self.cnn_config.out_channels}) must match patch embedding input channels ({self.patch_config.in_channels})"
        
        # Patch embedding output should match ViT input
        assert self.patch_config.embed_dim == self.vit_config.embed_dim, \
            f"Patch embedding dim ({self.patch_config.embed_dim}) must match ViT embedding dim ({self.vit_config.embed_dim})"
        
        # Data target size should work with CNN architecture
        expected_cnn_output = self.data_config.target_size // 8  # 3 pooling layers of 2x each
        assert expected_cnn_output == self.patch_config.feature_size, \
            f"Expected CNN output size ({expected_cnn_output}) must match patch feature size ({self.patch_config.feature_size})"
        
        # Number of classes should be consistent
        assert self.num_classes == self.vit_config.num_classes, \
            f"Model num_classes ({self.num_classes}) must match ViT num_classes ({self.vit_config.num_classes})"
    
    def get_num_patches(self):
        """Calculate number of patches"""
        patches_per_dim = self.patch_config.feature_size // self.patch_config.patch_size
        return patches_per_dim ** 2
    
    def get_sequence_length(self):
        """Get sequence length (patches + CLS token)"""
        return self.get_num_patches() + 1  # +1 for CLS token
    
    def print_config_summary(self):
        """Print configuration summary"""
        print("🔧 Model Configuration Summary")
        print("=" * 50)
        print(f"Model: {self.model_name} {self.model_version}")
        print(f"Task: {self.task}")
        print(f"Number of classes: {self.num_classes}")
        print()
        
        print("📊 Data Configuration:")
        print(f"  Input size: {self.data_config.target_size}x{self.data_config.target_size}")
        print(f"  Batch size: {self.data_config.batch_size}")
        print(f"  Min samples per class: {self.data_config.min_samples_per_class}")
        print()
        
        print("🏗️ CNN Backbone:")
        print(f"  Input: {self.cnn_config.in_channels} channels")
        print(f"  Output: {self.cnn_config.out_channels} channels")
        print(f"  Architecture: {self.cnn_config.conv1_out_channels} → {self.cnn_config.conv2_out_channels} → {self.cnn_config.conv3_out_channels}")
        print()
        
        print("🔗 Patch Embedding:")
        print(f"  Feature size: {self.patch_config.feature_size}x{self.patch_config.feature_size}")
        print(f"  Patch size: {self.patch_config.patch_size}x{self.patch_config.patch_size}")
        print(f"  Number of patches: {self.get_num_patches()}")
        print(f"  Embedding dim: {self.patch_config.embed_dim}")
        print()
        
        print("🤖 Vision Transformer:")
        print(f"  Embedding dim: {self.vit_config.embed_dim}")
        print(f"  Number of layers: {self.vit_config.num_layers}")
        print(f"  Number of heads: {self.vit_config.num_heads}")
        print(f"  MLP ratio: {self.vit_config.mlp_ratio}")
        print(f"  Sequence length: {self.get_sequence_length()} (patches + CLS)")
        print()
        
        print("⚙️ Training Settings:")
        print(f"  Device: {self.device}")
        print(f"  Mixed precision: {self.mixed_precision}")
        print(f"  Dropout rate: {self.vit_config.dropout_rate}")
        print("=" * 50)


class ConfigManager:
    """Utility class for managing configurations"""
    
    @staticmethod
    def save_config(config: HybridModelConfig, filepath: str):
        """Save configuration to JSON file"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Convert to dictionary
        config_dict = asdict(config)
        
        # Save to JSON
        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        print(f"✅ Configuration saved to {filepath}")
    
    @staticmethod
    def load_config(filepath: str) -> HybridModelConfig:
        """Load configuration from JSON file"""
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        
        # Reconstruct nested configs
        config = ConfigManager._dict_to_config(config_dict)
        print(f"✅ Configuration loaded from {filepath}")
        return config
    
    @staticmethod
    def _dict_to_config(config_dict: Dict[str, Any]) -> HybridModelConfig:
        """Convert dictionary back to config objects"""
        # Extract sub-configs
        data_config = DataConfig(**config_dict.pop('data_config', {}))
        cnn_config = CNNBackboneConfig(**config_dict.pop('cnn_config', {}))
        patch_config = PatchEmbeddingConfig(**config_dict.pop('patch_config', {}))
        
        # Handle nested ViT config
        vit_dict = config_dict.pop('vit_config', {})
        
        # Extract nested configs from ViT
        transformer_dict = vit_dict.pop('transformer_config', {})
        attention_dict = transformer_dict.pop('attention_config', {})
        mlp_dict = transformer_dict.pop('mlp_config', {})
        cls_token_dict = vit_dict.pop('cls_token_config', {})
        
        # Reconstruct ViT config
        attention_config = AttentionConfig(**attention_dict)
        mlp_config = MLPConfig(**mlp_dict)
        transformer_config = TransformerBlockConfig(
            attention_config=attention_config,
            mlp_config=mlp_config,
            **transformer_dict
        )
        cls_token_config = CLSTokenConfig(**cls_token_dict)
        vit_config = ViTConfig(
            transformer_config=transformer_config,
            cls_token_config=cls_token_config,
            **vit_dict
        )
        
        # Create main config
        config = HybridModelConfig(
            data_config=data_config,
            cnn_config=cnn_config,
            patch_config=patch_config,
            vit_config=vit_config,
            **config_dict
        )
        
        return config
    
    @staticmethod
    def create_default_config(num_classes: int = 10) -> HybridModelConfig:
        """Create default configuration"""
        return HybridModelConfig(num_classes=num_classes)
    
    @staticmethod
    def create_small_config(num_classes: int = 10) -> HybridModelConfig:
        """Create smaller model configuration for testing/quick training"""
        config = HybridModelConfig(
            model_name="HybridCNNViT-Small",
            num_classes=num_classes
        )
        
        # Smaller ViT
        config.vit_config.embed_dim = 256
        config.vit_config.num_layers = 4
        config.vit_config.num_heads = 4
        config.vit_config.mlp_ratio = 2.0
        
        # Update dependent configs
        config.patch_config.embed_dim = 256
        config.vit_config.transformer_config.embed_dim = 256
        config.vit_config.transformer_config.num_heads = 4
        config.vit_config.transformer_config.mlp_ratio = 2.0
        config.vit_config.cls_token_config.embed_dim = 256
        
        # Smaller CNN
        config.cnn_config.out_channels = 128
        config.patch_config.in_channels = 128
        
        # Smaller batch size
        config.data_config.batch_size = 16
        
        return config
    
    @staticmethod
    def create_large_config(num_classes: int = 10) -> HybridModelConfig:
        """Create larger model configuration for best performance"""
        config = HybridModelConfig(
            model_name="HybridCNNViT-Large",
            num_classes=num_classes
        )
        
        # Larger ViT
        config.vit_config.embed_dim = 768
        config.vit_config.num_layers = 12
        config.vit_config.num_heads = 12
        config.vit_config.mlp_ratio = 4.0
        
        # Update dependent configs
        config.patch_config.embed_dim = 768
        config.vit_config.transformer_config.embed_dim = 768
        config.vit_config.transformer_config.num_heads = 12
        config.vit_config.cls_token_config.embed_dim = 768
        
        # Larger CNN
        config.cnn_config.out_channels = 384
        config.patch_config.in_channels = 384
        
        return config


# Example usage and testing
def test_configuration():
    """Test configuration system"""
    print("🧪 Testing Configuration System...")
    
    # Create default config
    config = ConfigManager.create_default_config(num_classes=15)
    config.print_config_summary()
    
    print("\n📊 Configuration Details:")
    print(f"Number of patches: {config.get_num_patches()}")
    print(f"Sequence length: {config.get_sequence_length()}")
    
    # Test saving and loading
    test_path = "test_config.json"
    ConfigManager.save_config(config, test_path)
    loaded_config = ConfigManager.load_config(test_path)
    
    print("\n✅ Configuration system test passed!")
    
    # Clean up
    if os.path.exists(test_path):
        os.remove(test_path)
    
    return config


def show_config_variants():
    """Show different configuration variants"""
    print("🔧 Configuration Variants:")
    print("=" * 50)
    
    configs = [
        ("Small (Fast Training)", ConfigManager.create_small_config(10)),
        ("Default (Balanced)", ConfigManager.create_default_config(10)),
        ("Large (Best Performance)", ConfigManager.create_large_config(10))
    ]
    
    for name, config in configs:
        print(f"\n{name}:")
        print(f"  ViT: {config.vit_config.embed_dim}d, {config.vit_config.num_layers}L, {config.vit_config.num_heads}H")
        print(f"  CNN: {config.cnn_config.out_channels} channels")
        print(f"  Batch: {config.data_config.batch_size}")
        
        # Estimate parameters
        vit_params = config.vit_config.embed_dim ** 2 * config.vit_config.num_layers * 4  # Rough estimate
        print(f"  Est. ViT params: ~{vit_params/1e6:.1f}M")


if __name__ == "__main__":
    # Test the configuration system
    config = test_configuration()
    
    # Show variants
    show_config_variants()
    
    print("\n🎯 Ready to build model with configuration!")
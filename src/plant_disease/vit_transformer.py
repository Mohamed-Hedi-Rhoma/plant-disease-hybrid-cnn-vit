import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadSelfAttention(nn.Module):
    """
    Multi-Head Self-Attention mechanism for Vision Transformer
    
    Purpose: Allow patches to attend to each other and learn relationships
    Flow: [batch, 37, 512] → [batch, 37, 512] (37 = 36 patches + 1 CLS token)
    
    This is the core of ViT that enables:
    1. Each patch to look at all other patches
    2. Learning spatial relationships between disease symptoms
    3. Focusing on relevant parts for classification
    """
    
    def __init__(self, embed_dim=512, num_heads=8, dropout=0.1):
        """
        Initialize Multi-Head Self-Attention
        
        Args:
            embed_dim (int): Embedding dimension (512)
            num_heads (int): Number of attention heads (8)
            dropout (float): Dropout rate (0.1)
        """
        super(MultiHeadSelfAttention, self).__init__()
        
        # Store configuration
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        
        # Ensure embed_dim is divisible by num_heads
        assert embed_dim % num_heads == 0, f"embed_dim {embed_dim} must be divisible by num_heads {num_heads}"
        
        self.head_dim = embed_dim // num_heads  # 512 // 8 = 64
        self.scale = math.sqrt(self.head_dim)   # Scaling factor for dot-product attention
        
        # Linear projections for Query, Key, Value
        # Each projection maps [batch, seq_len, embed_dim] → [batch, seq_len, embed_dim]
        self.query_projection = nn.Linear(embed_dim, embed_dim, bias=True)
        self.key_projection = nn.Linear(embed_dim, embed_dim, bias=True)
        self.value_projection = nn.Linear(embed_dim, embed_dim, bias=True)
        
        # Output projection
        self.output_projection = nn.Linear(embed_dim, embed_dim, bias=True)
        
        # Dropout layers
        self.attention_dropout = nn.Dropout(dropout)
        self.output_dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """
        Initialize weights using Xavier uniform initialization
        Good for attention mechanisms
        """
        for module in [self.query_projection, self.key_projection, 
                      self.value_projection, self.output_projection]:
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        """
        Forward pass through multi-head self-attention
        
        Args:
            x (torch.Tensor): Input embeddings [batch_size, seq_len, embed_dim]
                             seq_len = 37 (36 patches + 1 CLS token)
            
        Returns:
            torch.Tensor: Attended embeddings [batch_size, seq_len, embed_dim]
            torch.Tensor: Attention weights [batch_size, num_heads, seq_len, seq_len] (for visualization)
        """
        batch_size, seq_len, embed_dim = x.shape
        
        # Input validation
        assert embed_dim == self.embed_dim, f"Expected embed_dim {self.embed_dim}, got {embed_dim}"
        
        # Step 1: Project to Query, Key, Value
        # Each: [batch_size, seq_len, embed_dim]
        Q = self.query_projection(x)  # What each patch is looking for
        K = self.key_projection(x)    # What each patch offers as key
        V = self.value_projection(x)  # What each patch offers as value
        
        # Step 2: Reshape for multi-head attention
        # [batch_size, seq_len, embed_dim] → [batch_size, seq_len, num_heads, head_dim]
        # → [batch_size, num_heads, seq_len, head_dim]
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Step 3: Compute attention scores
        # Q @ K^T: [batch_size, num_heads, seq_len, head_dim] @ [batch_size, num_heads, head_dim, seq_len]
        # Result: [batch_size, num_heads, seq_len, seq_len]
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        # Step 4: Apply softmax to get attention weights
        # Each row sums to 1 - represents how much each patch attends to others
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.attention_dropout(attention_weights)
        
        # Step 5: Apply attention to values
        # [batch_size, num_heads, seq_len, seq_len] @ [batch_size, num_heads, seq_len, head_dim]
        # Result: [batch_size, num_heads, seq_len, head_dim]
        attended_values = torch.matmul(attention_weights, V)
        
        # Step 6: Concatenate heads
        # [batch_size, num_heads, seq_len, head_dim] → [batch_size, seq_len, num_heads, head_dim]
        # → [batch_size, seq_len, embed_dim]
        attended_values = attended_values.transpose(1, 2).contiguous()
        attended_values = attended_values.view(batch_size, seq_len, embed_dim)
        
        # Step 7: Final output projection
        output = self.output_projection(attended_values)
        output = self.output_dropout(output)
        
        return output, attention_weights
    
    def print_model_info(self):
        """Print model architecture information"""
        print("🔍 Multi-Head Self-Attention Architecture:")
        print("=" * 50)
        print(f"Embedding dimension: {self.embed_dim}")
        print(f"Number of heads: {self.num_heads}")
        print(f"Head dimension: {self.head_dim}")
        print(f"Scaling factor: {self.scale:.3f}")
        print(f"Dropout rate: {self.dropout}")
        print("=" * 50)
        
        # Count parameters
        total_params = sum(p.numel() for p in self.parameters())
        print(f"Total parameters: {total_params:,}")


class FeedForward(nn.Module):
    """
    Feed-Forward Network (MLP) for Transformer
    
    Purpose: Process attended features through non-linear transformations
    Flow: [batch, seq_len, 512] → [batch, seq_len, 2048] → [batch, seq_len, 512]
    
    Standard transformer FFN:
    1. Expand dimension (usually 4x)
    2. Apply non-linearity (GELU)
    3. Contract back to original dimension
    """
    
    def __init__(self, embed_dim=512, hidden_dim=2048, dropout=0.1):
        """
        Initialize Feed-Forward Network
        
        Args:
            embed_dim (int): Input/output dimension (512)
            hidden_dim (int): Hidden dimension (2048, usually 4x embed_dim)
            dropout (float): Dropout rate (0.1)
        """
        super(FeedForward, self).__init__()
        
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        
        # Two linear layers with GELU activation
        self.linear1 = nn.Linear(embed_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, embed_dim)
        self.dropout_layer = nn.Dropout(dropout)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights using Xavier uniform"""
        for module in [self.linear1, self.linear2]:
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        """
        Forward pass through feed-forward network
        
        Args:
            x (torch.Tensor): Input [batch_size, seq_len, embed_dim]
            
        Returns:
            torch.Tensor: Output [batch_size, seq_len, embed_dim]
        """
        # Expand → Activate → Contract
        x = self.linear1(x)           # [batch, seq_len, hidden_dim]
        x = F.gelu(x)                 # GELU activation (smoother than ReLU)
        x = self.dropout_layer(x)     # Regularization
        x = self.linear2(x)           # [batch, seq_len, embed_dim]
        x = self.dropout_layer(x)     # Final dropout
        
        return x


class TransformerBlock(nn.Module):
    """
    Complete Transformer Block
    
    Purpose: Core building block of Vision Transformer
    Flow: x → LayerNorm → Self-Attention → Residual → LayerNorm → FFN → Residual
    
    This is the standard transformer block with:
    1. Pre-layer normalization (more stable than post-norm)
    2. Residual connections for gradient flow
    3. Multi-head self-attention for patch interactions
    4. Feed-forward network for feature processing
    """
    
    def __init__(self, embed_dim=512, num_heads=8, mlp_ratio=4.0, dropout=0.1):
        """
        Initialize Transformer Block
        
        Args:
            embed_dim (int): Embedding dimension (512)
            num_heads (int): Number of attention heads (8)
            mlp_ratio (float): Ratio for MLP hidden dimension (4.0)
            dropout (float): Dropout rate (0.1)
        """
        super(TransformerBlock, self).__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.dropout = dropout
        
        # Calculate MLP hidden dimension
        mlp_hidden_dim = int(embed_dim * mlp_ratio)  # 512 * 4 = 2048
        
        # Layer normalization (applied before attention and MLP)
        self.norm1 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.norm2 = nn.LayerNorm(embed_dim, eps=1e-6)
        
        # Multi-head self-attention
        self.attention = MultiHeadSelfAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # Feed-forward network
        self.mlp = FeedForward(
            embed_dim=embed_dim,
            hidden_dim=mlp_hidden_dim,
            dropout=dropout
        )
    
    def forward(self, x):
        """
        Forward pass through transformer block
        
        Args:
            x (torch.Tensor): Input embeddings [batch_size, seq_len, embed_dim]
            
        Returns:
            torch.Tensor: Output embeddings [batch_size, seq_len, embed_dim]
            torch.Tensor: Attention weights [batch_size, num_heads, seq_len, seq_len]
        """
        # Store residual
        residual1 = x
        
        # Pre-norm + Self-attention + Residual
        x_norm1 = self.norm1(x)
        attention_output, attention_weights = self.attention(x_norm1)
        x = residual1 + attention_output  # Residual connection
        
        # Store residual
        residual2 = x
        
        # Pre-norm + MLP + Residual
        x_norm2 = self.norm2(x)
        mlp_output = self.mlp(x_norm2)
        x = residual2 + mlp_output  # Residual connection
        
        return x, attention_weights
    
    def print_model_info(self):
        """Print model architecture information"""
        print("🏗️ Transformer Block Architecture:")
        print("=" * 50)
        print(f"Embedding dimension: {self.embed_dim}")
        print(f"Number of heads: {self.num_heads}")
        print(f"MLP ratio: {self.mlp_ratio}")
        print(f"MLP hidden dim: {int(self.embed_dim * self.mlp_ratio)}")
        print(f"Dropout rate: {self.dropout}")
        print("=" * 50)
        
        # Count parameters
        total_params = sum(p.numel() for p in self.parameters())
        print(f"Total parameters: {total_params:,}")


class ClassificationToken(nn.Module):
    """
    Classification Token (CLS Token) for ViT
    
    Purpose: Learnable token that aggregates information from all patches
    Flow: Adds one token to sequence [batch, 36, 512] → [batch, 37, 512]
    
    The CLS token:
    1. Starts as a learnable embedding
    2. Attends to all patches through self-attention
    3. Gets updated with global image information
    4. Used for final classification (like global average pooling)
    """
    
    def __init__(self, embed_dim=512):
        """
        Initialize Classification Token
        
        Args:
            embed_dim (int): Embedding dimension (512)
        """
        super(ClassificationToken, self).__init__()
        
        self.embed_dim = embed_dim
        
        # Learnable CLS token - one token of embed_dim size
        # Will be expanded to match batch size during forward pass
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        
        # Initialize with small random values
        nn.init.trunc_normal_(self.cls_token, std=0.02)
    
    def forward(self, x):
        """
        Add CLS token to patch embeddings
        
        Args:
            x (torch.Tensor): Patch embeddings [batch_size, num_patches, embed_dim]
                             e.g., [32, 36, 512]
        
        Returns:
            torch.Tensor: Embeddings with CLS token [batch_size, num_patches+1, embed_dim]
                         e.g., [32, 37, 512]
        """
        batch_size = x.size(0)
        
        # Expand CLS token to match batch size
        # [1, 1, embed_dim] → [batch_size, 1, embed_dim]
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        
        # Concatenate CLS token with patch embeddings
        # CLS token goes first: [CLS, patch1, patch2, ..., patch36]
        x = torch.cat([cls_tokens, x], dim=1)
        
        return x


# Testing functions
def test_attention():
    """Test Multi-Head Self-Attention"""
    print("🧪 Testing Multi-Head Self-Attention...")
    
    attention = MultiHeadSelfAttention(embed_dim=512, num_heads=8, dropout=0.1)
    attention.print_model_info()
    
    # Sample input (37 tokens: 1 CLS + 36 patches)
    batch_size = 32
    seq_len = 37  # 1 CLS + 36 patches
    embed_dim = 512
    
    sample_input = torch.randn(batch_size, seq_len, embed_dim)
    print(f"\nInput shape: {sample_input.shape}")
    
    # Forward pass
    attention.eval()
    with torch.no_grad():
        output, attn_weights = attention(sample_input)
    
    print(f"Output shape: {output.shape}")
    print(f"Attention weights shape: {attn_weights.shape}")
    print(f"Expected output: [32, 37, 512]")
    print(f"Expected attention: [32, 8, 37, 37]")
    
    assert output.shape == (batch_size, seq_len, embed_dim)
    assert attn_weights.shape == (batch_size, 8, seq_len, seq_len)
    print("✅ Multi-Head Self-Attention test passed!")


def test_transformer_block():
    """Test complete Transformer Block"""
    print("\n🧪 Testing Transformer Block...")
    
    transformer = TransformerBlock(embed_dim=512, num_heads=8, mlp_ratio=4.0, dropout=0.1)
    transformer.print_model_info()
    
    # Sample input
    batch_size = 32
    seq_len = 37
    embed_dim = 512
    
    sample_input = torch.randn(batch_size, seq_len, embed_dim)
    print(f"\nInput shape: {sample_input.shape}")
    
    # Forward pass
    transformer.eval()
    with torch.no_grad():
        output, attn_weights = transformer(sample_input)
    
    print(f"Output shape: {output.shape}")
    print(f"Attention weights shape: {attn_weights.shape}")
    
    assert output.shape == (batch_size, seq_len, embed_dim)
    print("✅ Transformer Block test passed!")


def test_cls_token():
    """Test Classification Token"""
    print("\n🧪 Testing Classification Token...")
    
    cls_token = ClassificationToken(embed_dim=512)
    
    # Sample patch embeddings (without CLS)
    batch_size = 32
    num_patches = 36
    embed_dim = 512
    
    patch_embeddings = torch.randn(batch_size, num_patches, embed_dim)
    print(f"Patch embeddings shape: {patch_embeddings.shape}")
    
    # Add CLS token
    with torch.no_grad():
        output = cls_token(patch_embeddings)
    
    print(f"Output with CLS token: {output.shape}")
    print(f"Expected: [32, 37, 512]")
    
    assert output.shape == (batch_size, num_patches + 1, embed_dim)
    print("✅ Classification Token test passed!")


def test_complete_pipeline():
    """Test complete pipeline: Patches → CLS → Transformer"""
    print("\n🔗 Testing Complete Pipeline...")
    
    # Create components
    cls_token = ClassificationToken(embed_dim=512)
    transformer = TransformerBlock(embed_dim=512, num_heads=8, mlp_ratio=4.0, dropout=0.1)
    
    # Sample patch embeddings (from previous pipeline)
    batch_size = 32
    patch_embeddings = torch.randn(batch_size, 36, 512)
    
    print(f"1. Patch embeddings: {patch_embeddings.shape}")
    
    # Add CLS token
    with torch.no_grad():
        embeddings_with_cls = cls_token(patch_embeddings)
        print(f"2. With CLS token: {embeddings_with_cls.shape}")
        
        # Through transformer
        transformer.eval()
        transformer_output, attn_weights = transformer(embeddings_with_cls)
        print(f"3. After transformer: {transformer_output.shape}")
        
        # Extract CLS token for classification
        cls_output = transformer_output[:, 0, :]  # First token is CLS
        print(f"4. CLS token output: {cls_output.shape}")
    
    print("✅ Complete pipeline test passed!")
    print("🎯 Ready for classification head!")


if __name__ == "__main__":
    test_attention()
    test_transformer_block() 
    test_cls_token()
    test_complete_pipeline()
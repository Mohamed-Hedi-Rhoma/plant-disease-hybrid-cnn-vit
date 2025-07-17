import os
import sys
import time
import json
from pathlib import Path
from datetime import datetime
import logging

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Import our custom modules
from plant_disease.config import HybridModelConfig, ConfigManager
from plant_disease.pipeline_data import PlantDiseasePreprocessor
from plant_disease.cnn_backbone import CNNBackbone
from plant_disease.patch_embedding import PatchEmbedding 
from plant_disease.vit_transformer import MultiHeadSelfAttention, TransformerBlock,ClassificationToken
from tqdm import tqdm
# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class HybridCNNViT(nn.Module):
    """
    Complete Hybrid CNN + Vision Transformer for Plant Disease Classification
    
    Architecture Flow:
    Raw Images → CNN Backbone → Patch Embedding → CLS Token → 
    Transformer Stack → Classification Head → Disease Predictions
    """
    
    def __init__(self, config: HybridModelConfig):
        """
        Initialize complete hybrid model
        
        Args:
            config (HybridModelConfig): Model configuration
        """
        super(HybridCNNViT, self).__init__()
        
        self.config = config
        self.num_classes = config.num_classes
        
        # 1. CNN Backbone
        self.cnn_backbone = CNNBackbone(
            in_channels=config.cnn_config.in_channels,
            out_channels=config.cnn_config.out_channels
        )
        
        # 2. Patch Embedding
        self.patch_embedding = PatchEmbedding(
            feature_size=config.patch_config.feature_size,
            patch_size=config.patch_config.patch_size,
            in_channels=config.patch_config.in_channels,
            embed_dim=config.patch_config.embed_dim
        )
        
        # 3. Classification Token
        self.cls_token = ClassificationToken(
            embed_dim=config.vit_config.embed_dim
        )
        
        # 4. Transformer Layers
        self.transformer_layers = nn.ModuleList([
            TransformerBlock(
                embed_dim=config.vit_config.embed_dim,
                num_heads=config.vit_config.num_heads,
                mlp_ratio=config.vit_config.mlp_ratio,
                dropout=config.vit_config.dropout_rate
            )
            for _ in range(config.vit_config.num_layers)
        ])
        
        # 5. Final Layer Normalization
        self.final_norm = nn.LayerNorm(
            config.vit_config.embed_dim,
            eps=1e-6
        )
        
        # 6. Classification Head
        self.classification_head = nn.Sequential(
            nn.Dropout(config.vit_config.classifier_dropout_rate),
            nn.Linear(config.vit_config.embed_dim, config.num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
        
        logger.info(f"🤖 HybridCNNViT initialized with {self.count_parameters():,} parameters")
    
    def _initialize_weights(self):
        """Initialize classification head weights"""
        for module in self.classification_head:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, x, return_attention=False):
        """
        Forward pass through complete model
        
        Args:
            x (torch.Tensor): Input images [batch_size, 3, 384, 384]
            return_attention (bool): Whether to return attention weights
            
        Returns:
            torch.Tensor: Class logits [batch_size, num_classes]
            list: Attention weights if return_attention=True
        """
        batch_size = x.size(0)
        
        # 1. CNN Feature Extraction
        # [batch, 3, 384, 384] → [batch, 256, 48, 48]
        cnn_features = self.cnn_backbone(x)
        
        # 2. Patch Embedding
        # [batch, 256, 48, 48] → [batch, 36, 512]
        patch_embeddings = self.patch_embedding(cnn_features)
        
        # 3. Add CLS Token
        # [batch, 36, 512] → [batch, 37, 512]
        embeddings_with_cls = self.cls_token(patch_embeddings)
        
        # 4. Through Transformer Layers
        x = embeddings_with_cls
        attention_weights = []
        
        for layer in self.transformer_layers:
            x, attn = layer(x)
            if return_attention:
                attention_weights.append(attn)
        
        # 5. Final Layer Norm
        x = self.final_norm(x)
        
        # 6. Extract CLS Token for Classification
        cls_token_output = x[:, 0, :]  # First token is CLS
        
        # 7. Classification Head
        logits = self.classification_head(cls_token_output)
        
        if return_attention:
            return logits, attention_weights
        else:
            return logits
    
    def count_parameters(self):
        """Count total trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def print_model_info(self):
        """Print detailed model information"""
        print("🤖 Hybrid CNN-ViT Model Information")
        print("=" * 60)
        
        # Component parameters
        cnn_params = sum(p.numel() for p in self.cnn_backbone.parameters() if p.requires_grad)
        patch_params = sum(p.numel() for p in self.patch_embedding.parameters() if p.requires_grad)
        cls_params = sum(p.numel() for p in self.cls_token.parameters() if p.requires_grad)
        transformer_params = sum(p.numel() for p in self.transformer_layers.parameters() if p.requires_grad)
        head_params = sum(p.numel() for p in self.classification_head.parameters() if p.requires_grad)
        
        print(f"CNN Backbone:        {cnn_params:,} parameters")
        print(f"Patch Embedding:     {patch_params:,} parameters")
        print(f"CLS Token:           {cls_params:,} parameters")
        print(f"Transformer Stack:   {transformer_params:,} parameters")
        print(f"Classification Head: {head_params:,} parameters")
        print("-" * 60)
        print(f"Total:               {self.count_parameters():,} parameters")
        print(f"Model Size:          {self.count_parameters() * 4 / 1024 / 1024:.2f} MB")
        print("=" * 60)


class TrainingManager:
    """
    Complete training manager for Hybrid CNN-ViT
    Handles training, validation, logging, and model saving
    """
    
    def __init__(self, config: HybridModelConfig):
        """
        Initialize training manager
        
        Args:
            config (HybridModelConfig): Model configuration
        """
        self.config = config
        self.device = self._setup_device()
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.scaler = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.preprocessing_info = None
        
        # Training state
        self.current_epoch = 0
        self.best_val_accuracy = 0.0
        self.training_history = {
            'train_loss': [],
            'train_accuracy': [],
            'val_loss': [],
            'val_accuracy': [],
            'learning_rate': []
        }
        
        # Create output directories
        self.output_dir = Path(config.save_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.plots_dir = self.output_dir / "plots"
        self.plots_dir.mkdir(exist_ok=True)
        
        logger.info(f"🚀 Training Manager initialized")
        logger.info(f"📁 Output directory: {self.output_dir}")
        logger.info(f"🔧 Device: {self.device}")
    
    def _setup_device(self):
        """Setup training device"""
        if self.config.device == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(self.config.device)
        
        if device.type == "cuda":
            logger.info(f"🔥 Using GPU: {torch.cuda.get_device_name()}")
            logger.info(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        else:
            logger.info("💻 Using CPU")
        
        return device
    
    def setup_data(self):
        """Setup data loaders using our preprocessing pipeline"""
        logger.info("📊 Setting up data loaders...")
        
        # Initialize preprocessor with config
        preprocessor = PlantDiseasePreprocessor(
            data_dir=self.config.data_config.data_dir,
            target_size=self.config.data_config.target_size,
            min_samples_per_class=self.config.data_config.min_samples_per_class,
            clean_augmented=self.config.data_config.clean_augmented
        )
        
        # Create data loaders
        self.train_loader, self.val_loader, self.test_loader, self.preprocessing_info = \
            preprocessor.create_dataloaders(
                batch_size=self.config.data_config.batch_size,
                num_workers=self.config.data_config.num_workers,
                use_weighted_sampling=self.config.data_config.use_weighted_sampling
            )
        
        # Update config with actual number of classes
        actual_num_classes = self.preprocessing_info['num_classes']
        self.config.num_classes = actual_num_classes
        self.config.vit_config.num_classes = actual_num_classes
        
        logger.info(f"📊 Data setup complete:")
        logger.info(f"   Classes: {actual_num_classes}")
        logger.info(f"   Train batches: {len(self.train_loader)}")
        logger.info(f"   Val batches: {len(self.val_loader)}")
        logger.info(f"   Test batches: {len(self.test_loader)}")
        
        # Save class mappings
        class_info_path = self.output_dir / "class_mappings.json"
        with open(class_info_path, 'w') as f:
            json.dump({
                'class_to_idx': self.preprocessing_info['class_to_idx'],
                'idx_to_class': self.preprocessing_info['idx_to_class']
            }, f, indent=2)
        
        logger.info(f"💾 Class mappings saved to {class_info_path}")
    
    def setup_model(self):
        """Setup model, optimizer, and training components"""
        logger.info("🤖 Setting up model...")
        
        # Create model
        self.model = HybridCNNViT(self.config).to(self.device)
        self.model.print_model_info()
        
        # Setup optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=0.001,
            weight_decay=0.01,
            betas=(0.9, 0.999)
        )
        
        # Setup learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=50,  # Will be updated based on actual epochs
            eta_min=1e-6
        )
        
        # Setup mixed precision training
        if self.config.mixed_precision and self.device.type == "cuda":
            self.scaler = GradScaler()
            logger.info("⚡ Mixed precision training enabled")
        
        # Compile model for speed (PyTorch 2.0+)
        if self.config.compile_model and hasattr(torch, 'compile'):
            self.model = torch.compile(self.model)
            logger.info("🚀 Model compiled for speed")
        
        logger.info("✅ Model setup complete")
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        # Use tqdm for progress bar
        progress_bar = tqdm(enumerate(self.train_loader), 
                        total=len(self.train_loader),
                        desc=f"Epoch {self.current_epoch+1} [Train]",
                        leave=False)
        
        for batch_idx, (images, labels) in progress_bar:
            images, labels = images.to(self.device), labels.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass with mixed precision
            if self.scaler is not None:
                with autocast():
                    logits = self.model(images)
                    loss = nn.CrossEntropyLoss()(logits, labels)
                
                # Backward pass
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                logits = self.model(images)
                loss = nn.CrossEntropyLoss()(logits, labels)
                loss.backward()
                self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            predictions = torch.argmax(logits, dim=1)
            correct_predictions += (predictions == labels).sum().item()
            total_samples += labels.size(0)
            
            # Update progress bar
            current_loss = total_loss / (batch_idx + 1)
            current_accuracy = correct_predictions / total_samples
            progress_bar.set_postfix({
                'loss': f'{current_loss:.4f}',
                'acc': f'{current_accuracy:.4f}',
                'lr': f'{self.scheduler.get_last_lr()[0]:.6f}'
            })
        
        # Calculate epoch metrics
        avg_loss = total_loss / len(self.train_loader)
        accuracy = correct_predictions / total_samples
        
        return avg_loss, accuracy
    
    def validate_epoch(self):
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_labels = []
        
        # Use tqdm for progress bar
        progress_bar = tqdm(enumerate(self.val_loader),
                        total=len(self.val_loader),
                        desc=f"Epoch {self.current_epoch+1} [Val]",
                        leave=False)
        
        with torch.no_grad():
            for batch_idx, (images, labels) in progress_bar:
                images, labels = images.to(self.device), labels.to(self.device)
                
                # Forward pass
                if self.scaler is not None:
                    with autocast():
                        logits = self.model(images)
                        loss = nn.CrossEntropyLoss()(logits, labels)
                else:
                    logits = self.model(images)
                    loss = nn.CrossEntropyLoss()(logits, labels)
                
                total_loss += loss.item()
                predictions = torch.argmax(logits, dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                # Update progress bar
                current_loss = total_loss / (batch_idx + 1)
                progress_bar.set_postfix({'loss': f'{current_loss:.4f}'})
        
        # Calculate metrics
        avg_loss = total_loss / len(self.val_loader)
        accuracy = accuracy_score(all_labels, all_predictions)
        
        return avg_loss, accuracy, all_predictions, all_labels
    
    def train(self, num_epochs=50):
        """
        Complete training loop
        
        Args:
            num_epochs (int): Number of training epochs
        """
        logger.info(f"🚀 Starting training for {num_epochs} epochs")
        
        # Update scheduler
        self.scheduler.T_max = num_epochs
        
        start_time = time.time()
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            epoch_start_time = time.time()
            
            logger.info(f"\n📈 Epoch {epoch+1}/{num_epochs}")
            logger.info("-" * 50)
            
            # Training
            train_loss, train_accuracy = self.train_epoch()
            
            # Validation
            val_loss, val_accuracy, val_predictions, val_labels = self.validate_epoch()
            
            # Update learning rate
            self.scheduler.step()
            current_lr = self.scheduler.get_last_lr()[0]
            
            # Record history
            self.training_history['train_loss'].append(train_loss)
            self.training_history['train_accuracy'].append(train_accuracy)
            self.training_history['val_loss'].append(val_loss)
            self.training_history['val_accuracy'].append(val_accuracy)
            self.training_history['learning_rate'].append(current_lr)
            
            # Calculate epoch time
            epoch_time = time.time() - epoch_start_time
            
            # Log epoch results
            logger.info(f"📊 Epoch {epoch+1} Results:")
            logger.info(f"   Train - Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.4f}")
            logger.info(f"   Val   - Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}")
            logger.info(f"   Learning Rate: {current_lr:.6f}")
            logger.info(f"   Time: {epoch_time:.2f}s")
            
            # Save best model
            if val_accuracy > self.best_val_accuracy:
                self.best_val_accuracy = val_accuracy
                self.save_checkpoint(is_best=True)
                logger.info(f"🎯 New best validation accuracy: {val_accuracy:.4f}")
            
            # Save regular checkpoint every 10 epochs
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(is_best=False)
            
            # Plot training progress
            if (epoch + 1) % 5 == 0:
                self.plot_training_progress()
        
        # Training complete
        total_time = time.time() - start_time
        logger.info(f"\n🎉 Training completed in {total_time/3600:.2f} hours")
        logger.info(f"🏆 Best validation accuracy: {self.best_val_accuracy:.4f}")
        
        # Final evaluation
        self.evaluate_model()
        
        # Save final training history
        self.save_training_history()
    
    def save_checkpoint(self, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_accuracy': self.best_val_accuracy,
            'training_history': self.training_history,
            'config': self.config,
            'preprocessing_info': self.preprocessing_info
        }
        
        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        # Save checkpoint
        checkpoint_name = "best_model.pt" if is_best else f"checkpoint_epoch_{self.current_epoch+1}.pt"
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        
        torch.save(checkpoint, checkpoint_path)
        
        if is_best:
            logger.info(f"💾 Best model saved to {checkpoint_path}")
    
    def plot_training_progress(self):
        """Plot training progress"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        epochs = range(1, len(self.training_history['train_loss']) + 1)
        
        # Loss plot
        axes[0, 0].plot(epochs, self.training_history['train_loss'], 'b-', label='Training Loss')
        axes[0, 0].plot(epochs, self.training_history['val_loss'], 'r-', label='Validation Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Accuracy plot
        axes[0, 1].plot(epochs, self.training_history['train_accuracy'], 'b-', label='Training Accuracy')
        axes[0, 1].plot(epochs, self.training_history['val_accuracy'], 'r-', label='Validation Accuracy')
        axes[0, 1].set_title('Training and Validation Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Learning rate plot
        axes[1, 0].plot(epochs, self.training_history['learning_rate'], 'g-')
        axes[1, 0].set_title('Learning Rate Schedule')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True)
        
        # Best accuracy so far
        axes[1, 1].axhline(y=self.best_val_accuracy, color='r', linestyle='--', 
                          label=f'Best Val Acc: {self.best_val_accuracy:.4f}')
        axes[1, 1].plot(epochs, self.training_history['val_accuracy'], 'b-', 
                       label='Validation Accuracy')
        axes[1, 1].set_title('Best Validation Accuracy Progress')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Accuracy')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.plots_dir / f"training_progress_epoch_{self.current_epoch+1}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def evaluate_model(self):
        """Evaluate model on test set"""
        logger.info("🧪 Evaluating model on test set...")
        
        self.model.eval()
        all_predictions = []
        all_labels = []
        
        # Use tqdm for progress bar
        progress_bar = tqdm(enumerate(self.test_loader),
                        total=len(self.test_loader),
                        desc="Evaluating",
                        leave=True)
        
        with torch.no_grad():
            for batch_idx, (images, labels) in progress_bar:
                images, labels = images.to(self.device), labels.to(self.device)
                
                logits = self.model(images)
                predictions = torch.argmax(logits, dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_predictions, average='weighted'
        )
        
        logger.info(f"📊 Test Set Results:")
        logger.info(f"   Accuracy:  {accuracy:.4f}")
        logger.info(f"   Precision: {precision:.4f}")
        logger.info(f"   Recall:    {recall:.4f}")
        logger.info(f"   F1 Score:  {f1:.4f}")
        
        # Confusion matrix
        self.plot_confusion_matrix(all_labels, all_predictions)
        
        return accuracy, precision, recall, f1
    
    def plot_confusion_matrix(self, labels, predictions):
        """Plot confusion matrix"""
        cm = confusion_matrix(labels, predictions)
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=list(self.preprocessing_info['idx_to_class'].values()),
                   yticklabels=list(self.preprocessing_info['idx_to_class'].values()))
        plt.title('Confusion Matrix - Test Set')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        
        # Save plot
        cm_path = self.plots_dir / "confusion_matrix.png"
        plt.savefig(cm_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"📊 Confusion matrix saved to {cm_path}")
    
    def save_training_history(self):
        """Save training history to JSON"""
        history_path = self.output_dir / "training_history.json"
        with open(history_path, 'w') as f:
            json.dump(self.training_history, f, indent=2)
        
        logger.info(f"📊 Training history saved to {history_path}")


def main():
    """Main training function"""
    print("🚀 Plant Disease Classification - Hybrid CNN-ViT Training")
    print("=" * 60)
    
    # Create configuration
    config = ConfigManager.create_default_config(num_classes=15)  # Will be updated from data
    config.print_config_summary()
    
    # Save configuration
    config_path = "./training_config.json"
    ConfigManager.save_config(config, config_path)
    
    # Initialize training manager
    trainer = TrainingManager(config)
    
    # Setup data
    trainer.setup_data()
    
    # Setup model
    trainer.setup_model()
    
    # Start training
    trainer.train(num_epochs=50)
    
    print("\n🎉 Training completed successfully!")
    print(f"📁 Results saved in: {trainer.output_dir}")


if __name__ == "__main__":
    main()
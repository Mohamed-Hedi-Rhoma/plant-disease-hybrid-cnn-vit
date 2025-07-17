import os
import json
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from sklearn.model_selection import train_test_split
from collections import Counter, defaultdict
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PlantDiseaseDataset(Dataset):
    """Custom Dataset for Plant Disease Images"""
    
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image
        image_path = self.image_paths[idx]
        image = Image.open(image_path)
        
        # Convert RGBA to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        # Get label
        label = self.labels[idx]
        
        return image, label

class PlantDiseasePreprocessor:
    """Complete preprocessing pipeline for plant disease detection"""
    
    def __init__(self, data_dir="ai_training_data", target_size=384, min_samples_per_class=1500, clean_augmented=True):
        self.data_dir = data_dir
        self.target_size = target_size
        self.min_samples_per_class = min_samples_per_class
        self.clean_augmented = clean_augmented
        
        # Will be calculated
        self.dataset_stats = None
        self.class_to_idx = {}
        self.idx_to_class = {}
        
        logger.info(f"🚀 Initializing preprocessing pipeline")
        logger.info(f"📁 Data directory: {data_dir}")
        logger.info(f"🎯 Target image size: {target_size}x{target_size}")
        logger.info(f"📊 Minimum samples per class: {min_samples_per_class}")
        logger.info(f"🧹 Clean augmented files: {clean_augmented}")
    
    def clean_augmented_files(self):
        """Remove existing augmented files to prevent accumulation"""
        logger.info("🧹 Cleaning existing augmented files...")
        
        total_removed = 0
        
        # Get all disease directories
        disease_dirs = [d for d in os.listdir(self.data_dir) 
                       if os.path.isdir(os.path.join(self.data_dir, d))]
        
        for disease in disease_dirs:
            disease_path = os.path.join(self.data_dir, disease)
            
            # Find augmented files (containing '_aug_')
            all_files = os.listdir(disease_path)
            aug_files = [f for f in all_files if '_aug_' in f and f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            # Remove augmented files
            for aug_file in aug_files:
                try:
                    os.remove(os.path.join(disease_path, aug_file))
                    total_removed += 1
                except Exception as e:
                    logger.warning(f"Could not remove {aug_file}: {e}")
            
            if aug_files:
                logger.info(f"   {disease}: removed {len(aug_files)} augmented files")
        
        logger.info(f"🧹 Cleanup complete: removed {total_removed} augmented files")
        return total_removed
    
    def scan_dataset(self):
        """Scan dataset and create image paths and labels (original images only)"""
        logger.info("🔍 Scanning dataset...")
        
        image_paths = []
        labels = []
        class_counts = defaultdict(int)
        
        # Get all disease directories
        disease_dirs = [d for d in os.listdir(self.data_dir) 
                       if os.path.isdir(os.path.join(self.data_dir, d))]
        
        # Create class mappings
        self.class_to_idx = {disease: idx for idx, disease in enumerate(sorted(disease_dirs))}
        self.idx_to_class = {idx: disease for disease, idx in self.class_to_idx.items()}
        
        # Scan each disease directory (only original images, no augmented)
        for disease in disease_dirs:
            disease_path = os.path.join(self.data_dir, disease)
            
            # Get all images (exclude augmented files)
            image_files = [f for f in os.listdir(disease_path) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png')) and '_aug_' not in f]
            
            # Add to lists
            for img_file in image_files:
                img_path = os.path.join(disease_path, img_file)
                image_paths.append(img_path)
                labels.append(self.class_to_idx[disease])
                class_counts[disease] += 1
        
        logger.info(f"📊 Dataset scan complete (original images only):")
        logger.info(f"   Total images: {len(image_paths)}")
        logger.info(f"   Total classes: {len(disease_dirs)}")
        
        for disease, count in sorted(class_counts.items()):
            logger.info(f"   {disease}: {count} images")
        
        return image_paths, labels, class_counts
    
    def split_dataset(self, image_paths, labels):
        """Split dataset into train/val/test BEFORE augmentation"""
        logger.info("📊 Splitting dataset (before augmentation)...")
        
        # First split: Train + Val (85%) vs Test (15%)
        train_val_paths, test_paths, train_val_labels, test_labels = train_test_split(
            image_paths, labels, test_size=0.15, random_state=42, stratify=labels
        )
        
        # Second split: Train (70% of total) vs Val (15% of total)
        train_paths, val_paths, train_labels, val_labels = train_test_split(
            train_val_paths, train_val_labels, test_size=0.176, random_state=42, stratify=train_val_labels  # 0.176 * 0.85 ≈ 0.15
        )
        
        # Calculate class distribution for each split
        train_class_counts = Counter(train_labels)
        val_class_counts = Counter(val_labels)
        test_class_counts = Counter(test_labels)
        
        logger.info(f"📊 Data split (original images only):")
        logger.info(f"   Train: {len(train_paths)} images ({len(train_paths)/len(image_paths)*100:.1f}%)")
        logger.info(f"   Val: {len(val_paths)} images ({len(val_paths)/len(image_paths)*100:.1f}%)")
        logger.info(f"   Test: {len(test_paths)} images ({len(test_paths)/len(image_paths)*100:.1f}%)")
        
        # Show class distribution
        logger.info(f"📊 Class distribution in splits:")
        for disease_name, class_idx in sorted(self.class_to_idx.items()):
            train_count = train_class_counts.get(class_idx, 0)
            val_count = val_class_counts.get(class_idx, 0)
            test_count = test_class_counts.get(class_idx, 0)
            logger.info(f"   {disease_name}: Train={train_count}, Val={val_count}, Test={test_count}")
        
        return (train_paths, train_labels), (val_paths, val_labels), (test_paths, test_labels)
    
    def calculate_dataset_statistics(self, train_paths, train_labels):
        """Calculate dataset mean and std for normalization (using training set only)"""
        logger.info("📊 Calculating dataset statistics (from training set)...")
        
        # Use a subset for efficiency (max 1000 images)
        if len(train_paths) > 1000:
            indices = np.random.choice(len(train_paths), 1000, replace=False)
            sample_paths = [train_paths[i] for i in indices]
        else:
            sample_paths = train_paths
        
        # Calculate statistics
        transform = transforms.Compose([
            transforms.Resize((self.target_size, self.target_size)),
            transforms.ToTensor()
        ])
        
        mean = torch.zeros(3)
        std = torch.zeros(3)
        total_samples = 0
        
        for img_path in sample_paths:
            try:
                image = Image.open(img_path)
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                
                tensor = transform(image)
                mean += tensor.mean(dim=(1, 2))
                std += tensor.std(dim=(1, 2))
                total_samples += 1
                
            except Exception as e:
                logger.warning(f"Error processing {img_path}: {e}")
                continue
        
        mean /= total_samples
        std /= total_samples
        
        self.dataset_stats = {
            'mean': mean.tolist(),
            'std': std.tolist()
        }
        
        logger.info(f"📊 Dataset statistics calculated:")
        logger.info(f"   Mean: {mean.tolist()}")
        logger.info(f"   Std: {std.tolist()}")
        
        return self.dataset_stats
    
    def create_augmented_samples(self, train_paths, train_labels):
        """Create augmented samples for underrepresented classes (training set only)"""
        logger.info("🔄 Creating augmented samples for training set...")
        
        # Count classes in training set
        train_class_counts = Counter(train_labels)
        train_class_counts_by_name = {}
        for class_idx, count in train_class_counts.items():
            disease_name = self.idx_to_class[class_idx]
            train_class_counts_by_name[disease_name] = count
        
        # Augmentation transform
        augment_transform = transforms.Compose([
            transforms.Resize((self.target_size, self.target_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        ])
        
        augmented_paths = []
        augmented_labels = []
        
        for disease_name, count in train_class_counts_by_name.items():
            if count < self.min_samples_per_class:
                class_idx = self.class_to_idx[disease_name]
                needed = self.min_samples_per_class - count
                
                # Get training images for this class
                class_train_images = [path for path, label in zip(train_paths, train_labels) 
                                    if label == class_idx]
                
                logger.info(f"   {disease_name}: {count} → {self.min_samples_per_class} (+{needed} augmented)")
                
                # Create augmented samples
                created_count = 0
                for i in range(needed):
                    # Select random original image
                    original_path = np.random.choice(class_train_images)
                    
                    # Create unique filename
                    disease_dir = os.path.dirname(original_path)
                    base_name = os.path.splitext(os.path.basename(original_path))[0]
                    aug_name = f"{base_name}_aug_{i:04d}.jpg"
                    aug_path = os.path.join(disease_dir, aug_name)
                    
                    # Skip if file already exists
                    if os.path.exists(aug_path):
                        continue
                    
                    # Create augmented version
                    try:
                        image = Image.open(original_path)
                        if image.mode != 'RGB':
                            image = image.convert('RGB')
                        
                        # Apply augmentation
                        augmented = augment_transform(image)
                        
                        # Convert tensor back to PIL for saving
                        if isinstance(augmented, torch.Tensor):
                            augmented = transforms.ToPILImage()(augmented)
                        
                        augmented.save(aug_path, quality=95)
                        
                        augmented_paths.append(aug_path)
                        augmented_labels.append(class_idx)
                        created_count += 1
                        
                    except Exception as e:
                        logger.warning(f"Error creating augmented sample: {e}")
                        continue
                
                logger.info(f"   {disease_name}: created {created_count} augmented images")
        
        logger.info(f"🔄 Augmentation complete: +{len(augmented_paths)} images")
        return augmented_paths, augmented_labels
    
    def create_transforms(self):
        """Create train, validation, and test transforms"""
        
        # Training transforms (with augmentation)
        train_transforms = transforms.Compose([
            transforms.Resize((self.target_size, self.target_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.dataset_stats['mean'], 
                               std=self.dataset_stats['std'])
        ])
        
        # Validation and test transforms (no augmentation)
        val_test_transforms = transforms.Compose([
            transforms.Resize((self.target_size, self.target_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.dataset_stats['mean'], 
                               std=self.dataset_stats['std'])
        ])
        
        return train_transforms, val_test_transforms
    
    def create_weighted_sampler(self, labels):
        """Create weighted sampler for balanced training"""
        class_counts = Counter(labels)
        class_weights = {cls: 1.0 / count for cls, count in class_counts.items()}
        
        weights = [class_weights[label] for label in labels]
        sampler = WeightedRandomSampler(weights, len(weights))
        
        return sampler
    
    def create_dataloaders(self, batch_size=32, num_workers=4, use_weighted_sampling=True):
        """Create complete preprocessing pipeline and return DataLoaders"""
        
        logger.info("🚀 Starting complete preprocessing pipeline...")
        
        # Step 0: Clean augmented files if requested
        if self.clean_augmented:
            self.clean_augmented_files()
        
        # Step 1: Scan dataset (only original images)
        image_paths, labels, class_counts = self.scan_dataset()
        
        # Step 2: Split dataset BEFORE augmentation (original images only)
        (train_paths, train_labels), (val_paths, val_labels), (test_paths, test_labels) = self.split_dataset(image_paths, labels)
        
        # Step 3: Calculate dataset statistics (from training set only)
        self.calculate_dataset_statistics(train_paths, train_labels)
        
        # Step 4: Create augmented samples for training set only
        aug_paths, aug_labels = self.create_augmented_samples(train_paths, train_labels)
        
        # Step 5: Combine original training data with augmented data
        final_train_paths = train_paths + aug_paths
        final_train_labels = train_labels + aug_labels
        
        logger.info(f"📊 Final dataset sizes:")
        logger.info(f"   Train: {len(final_train_paths)} images (original: {len(train_paths)}, augmented: {len(aug_paths)})")
        logger.info(f"   Val: {len(val_paths)} images (original only)")
        logger.info(f"   Test: {len(test_paths)} images (original only)")
        
        # Step 6: Create transforms
        train_transforms, val_test_transforms = self.create_transforms()
        
        # Step 7: Create datasets
        train_dataset = PlantDiseaseDataset(final_train_paths, final_train_labels, train_transforms)
        val_dataset = PlantDiseaseDataset(val_paths, val_labels, val_test_transforms)
        test_dataset = PlantDiseaseDataset(test_paths, test_labels, val_test_transforms)
        
        # Step 8: Create weighted sampler for training
        train_sampler = None
        if use_weighted_sampling:
            train_sampler = self.create_weighted_sampler(final_train_labels)
        
        # Step 9: Create DataLoaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            sampler=train_sampler,
            shuffle=(train_sampler is None),
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available()
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available()
        )
        
        test_loader = DataLoader(
            test_dataset, 
            batch_size=batch_size, 
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available()
        )
        
        # Step 10: Show final class distribution
        final_train_class_counts = Counter(final_train_labels)
        val_class_counts = Counter(val_labels)
        test_class_counts = Counter(test_labels)
        
        logger.info(f"📊 Final class distribution:")
        for disease_name, class_idx in sorted(self.class_to_idx.items()):
            train_count = final_train_class_counts.get(class_idx, 0)
            val_count = val_class_counts.get(class_idx, 0)
            test_count = test_class_counts.get(class_idx, 0)
            logger.info(f"   {disease_name}: Train={train_count}, Val={val_count}, Test={test_count}")
        
        # Save preprocessing info
        preprocessing_info = {
            'num_classes': len(self.class_to_idx),
            'class_to_idx': self.class_to_idx,
            'idx_to_class': self.idx_to_class,
            'dataset_stats': self.dataset_stats,
            'target_size': self.target_size,
            'data_split': {
                'train': len(final_train_paths),
                'val': len(val_paths),
                'test': len(test_paths)
            },
            'augmentation_info': {
                'augmented_samples': len(aug_paths),
                'original_train_samples': len(train_paths)
            }
        }
        
        logger.info("✅ Preprocessing pipeline complete!")
        logger.info(f"📊 Ready for training with {len(self.class_to_idx)} classes")
        logger.info(f"📊 Val/Test contain ONLY original images!")
        
        return train_loader, val_loader, test_loader, preprocessing_info

# Usage example
def main():
    """Example usage of the preprocessing pipeline"""
    
    # Initialize preprocessor
    preprocessor = PlantDiseasePreprocessor(
        data_dir="/home/mrhouma/Documents/Plant_diseases_project/ai_training_data",
        target_size=384,
        min_samples_per_class=1500,
        clean_augmented=True  # Clean old augmented files
    )
    
    # Create DataLoaders
    train_loader, val_loader, test_loader, info = preprocessor.create_dataloaders(
        batch_size=32,
        num_workers=4,
        use_weighted_sampling=True
    )
    
    # Print information
    print(f"\n🎯 DataLoaders ready for ViT training!")
    print(f"📊 Dataset info:")
    print(f"   Classes: {info['num_classes']}")
    print(f"   Image size: {info['target_size']}x{info['target_size']}")
    print(f"   Train batches: {len(train_loader)}")
    print(f"   Val batches: {len(val_loader)}")
    print(f"   Test batches: {len(test_loader)}")
    
    # Test a batch
    for images, labels in train_loader:
        print(f"\n📊 Batch shape:")
        print(f"   Images: {images.shape}")  # [batch_size, 3, 384, 384]
        print(f"   Labels: {labels.shape}")  # [batch_size]
        print(f"   Data type: {images.dtype}")
        print(f"   Value range: {images.min():.3f} to {images.max():.3f}")
        break
    
    return train_loader, val_loader, test_loader, info

if __name__ == "__main__":
    main()
import os
import json
from PIL import Image
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt

class DatasetAnalyzer:
    def __init__(self, data_dir="ai_training_data"):
        self.data_dir = data_dir
        self.stats = defaultdict(list)
        self.disease_stats = defaultdict(dict)
        
    def analyze_dataset(self):
        """Analyze the complete dataset"""
        print("🔍 Analyzing Plant Disease Dataset...")
        print("=" * 50)
        
        # Get all disease directories
        disease_dirs = [d for d in os.listdir(self.data_dir) 
                       if os.path.isdir(os.path.join(self.data_dir, d))]
        
        total_images = 0
        total_size_mb = 0
        
        for disease in disease_dirs:
            disease_path = os.path.join(self.data_dir, disease)
            
            # Get all images in disease folder
            image_files = [f for f in os.listdir(disease_path) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            if not image_files:
                continue
                
            print(f"\n📊 Analyzing {disease}...")
            
            disease_images = len(image_files)
            widths, heights, file_sizes = [], [], []
            
            # Analyze each image
            for img_file in image_files:
                img_path = os.path.join(disease_path, img_file)
                
                try:
                    # Get file size
                    file_size = os.path.getsize(img_path)
                    file_sizes.append(file_size)
                    
                    # Get image dimensions
                    with Image.open(img_path) as img:
                        width, height = img.size
                        widths.append(width)
                        heights.append(height)
                        
                except Exception as e:
                    print(f"   ❌ Error reading {img_file}: {e}")
                    continue
            
            # Calculate statistics for this disease
            if widths and heights:
                disease_size_mb = sum(file_sizes) / (1024 * 1024)
                
                self.disease_stats[disease] = {
                    'count': disease_images,
                    'size_mb': disease_size_mb,
                    'width_stats': {
                        'min': min(widths),
                        'max': max(widths),
                        'mean': np.mean(widths),
                        'median': np.median(widths)
                    },
                    'height_stats': {
                        'min': min(heights),
                        'max': max(heights),
                        'mean': np.mean(heights),
                        'median': np.median(heights)
                    },
                    'file_size_stats': {
                        'min_kb': min(file_sizes) / 1024,
                        'max_kb': max(file_sizes) / 1024,
                        'mean_kb': np.mean(file_sizes) / 1024,
                        'median_kb': np.median(file_sizes) / 1024
                    }
                }
                
                # Add to overall stats
                self.stats['widths'].extend(widths)
                self.stats['heights'].extend(heights)
                self.stats['file_sizes'].extend(file_sizes)
                
                total_images += disease_images
                total_size_mb += disease_size_mb
                
                print(f"   Images: {disease_images}")
                print(f"   Size: {disease_size_mb:.1f}MB")
                print(f"   Dimensions: {min(widths)}x{min(heights)} to {max(widths)}x{max(heights)}")
                print(f"   Avg size: {np.mean(file_sizes)/1024:.1f}KB")
        
        # Print overall summary
        print(f"\n" + "=" * 50)
        print("📈 DATASET SUMMARY")
        print("=" * 50)
        print(f"Total diseases: {len(disease_dirs)}")
        print(f"Total images: {total_images}")
        print(f"Total size: {total_size_mb:.1f}MB")
        
        if self.stats['widths']:
            print(f"Image width range: {min(self.stats['widths'])} - {max(self.stats['widths'])}")
            print(f"Image height range: {min(self.stats['heights'])} - {max(self.stats['heights'])}")
            print(f"Average image size: {np.mean(self.stats['file_sizes'])/1024:.1f}KB")
        
        return self.disease_stats
    
    def print_class_balance(self):
        """Print class balance information"""
        print(f"\n📊 CLASS BALANCE")
        print("-" * 30)
        
        # Sort by image count
        sorted_diseases = sorted(self.disease_stats.items(), 
                               key=lambda x: x[1]['count'], 
                               reverse=True)
        
        for disease, stats in sorted_diseases:
            print(f"{disease:25} {stats['count']:4d} images ({stats['size_mb']:5.1f}MB)")
    
    def check_image_quality(self, sample_size=5):
        """Check for potential image quality issues"""
        print(f"\n🔍 IMAGE QUALITY CHECK (sampling {sample_size} per disease)")
        print("-" * 50)
        
        for disease in self.disease_stats:
            disease_path = os.path.join(self.data_dir, disease)
            image_files = [f for f in os.listdir(disease_path) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            # Sample random images
            sample_files = np.random.choice(image_files, 
                                          min(sample_size, len(image_files)), 
                                          replace=False)
            
            issues = []
            for img_file in sample_files:
                img_path = os.path.join(disease_path, img_file)
                
                try:
                    with Image.open(img_path) as img:
                        width, height = img.size
                        
                        # Check for very small images
                        if width < 100 or height < 100:
                            issues.append(f"Very small: {img_file} ({width}x{height})")
                        
                        # Check for very large images
                        if width > 5000 or height > 5000:
                            issues.append(f"Very large: {img_file} ({width}x{height})")
                        
                        # Check aspect ratio
                        aspect_ratio = width / height
                        if aspect_ratio > 3 or aspect_ratio < 0.33:
                            issues.append(f"Extreme aspect ratio: {img_file} ({aspect_ratio:.2f})")
                            
                except Exception as e:
                    issues.append(f"Corrupted: {img_file} - {e}")
            
            if issues:
                print(f"\n❌ {disease}: {len(issues)} issues found")
                for issue in issues[:3]:  # Show first 3 issues
                    print(f"   {issue}")
            else:
                print(f"✅ {disease}: No issues in sample")
    
    def recommend_preprocessing(self):
        """Recommend preprocessing steps for ViT"""
        print(f"\n🚀 PREPROCESSING RECOMMENDATIONS FOR VISION TRANSFORMER")
        print("=" * 60)
        
        if not self.stats['widths']:
            print("No image data analyzed yet!")
            return
        
        # Calculate statistics
        widths = self.stats['widths']
        heights = self.stats['heights']
        
        median_width = np.median(widths)
        median_height = np.median(heights)
        
        print(f"📏 Current image sizes:")
        print(f"   Width: {min(widths)} - {max(widths)} (median: {median_width:.0f})")
        print(f"   Height: {min(heights)} - {max(heights)} (median: {median_height:.0f})")
        
        print(f"\n💡 Recommendations:")
        print(f"   1. Target size: 224x224 or 384x384 (standard for ViT)")
        print(f"   2. Resize strategy: Center crop + resize (maintains aspect ratio)")
        print(f"   3. Data augmentation: RandomResizedCrop, ColorJitter, RandomHorizontalFlip")
        print(f"   4. Normalization: ImageNet stats (mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])")
        
        # Check class balance
        counts = [stats['count'] for stats in self.disease_stats.values()]
        if len(set(counts)) > 1:
            print(f"   5. Class balancing: Consider weighted sampling (imbalanced classes detected)")
        
        print(f"\n📊 Estimated preprocessing time: ~{len(widths)*0.1:.1f} seconds")


def main():
    analyzer = DatasetAnalyzer()
    
    # Run analysis
    analyzer.analyze_dataset()
    analyzer.print_class_balance()
    analyzer.check_image_quality()
    analyzer.recommend_preprocessing()


if __name__ == "__main__":
    main()
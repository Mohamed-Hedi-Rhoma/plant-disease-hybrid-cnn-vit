import os
from PIL import Image
from collections import defaultdict
import numpy as np

def analyze_color_modes(data_dir="ai_training_data"):
    """Analyze color modes of all images in dataset"""
    print("🎨 Analyzing Image Color Modes...")
    print("=" * 50)
    
    # Track color modes
    mode_counts = defaultdict(int)
    disease_modes = defaultdict(lambda: defaultdict(int))
    issues = []
    
    # Get all disease directories
    disease_dirs = [d for d in os.listdir(data_dir) 
                   if os.path.isdir(os.path.join(data_dir, d))]
    
    total_images = 0
    
    for disease in disease_dirs:
        disease_path = os.path.join(data_dir, disease)
        
        # Get all images
        image_files = [f for f in os.listdir(disease_path) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        if not image_files:
            continue
            
        print(f"\n📊 {disease}: {len(image_files)} images")
        
        for img_file in image_files:
            img_path = os.path.join(disease_path, img_file)
            
            try:
                with Image.open(img_path) as img:
                    mode = img.mode
                    bands = len(img.getbands())
                    
                    # Count modes
                    mode_counts[mode] += 1
                    disease_modes[disease][mode] += 1
                    
                    # Check for potential issues
                    if mode not in ['RGB', 'L']:
                        issues.append(f"{disease}/{img_file}: {mode} (bands: {bands})")
                    
                    total_images += 1
                    
            except Exception as e:
                issues.append(f"{disease}/{img_file}: ERROR - {e}")
    
    # Print overall results
    print(f"\n" + "=" * 50)
    print("🎨 COLOR MODE SUMMARY")
    print("=" * 50)
    print(f"Total images analyzed: {total_images}")
    print(f"\nColor mode distribution:")
    
    for mode, count in sorted(mode_counts.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / total_images) * 100
        print(f"  {mode:8} {count:5d} images ({percentage:5.1f}%)")
    
    # Print per-disease breakdown
    print(f"\n📊 PER-DISEASE COLOR MODES:")
    print("-" * 30)
    
    for disease in sorted(disease_modes.keys()):
        modes = disease_modes[disease]
        total_disease = sum(modes.values())
        
        mode_str = ", ".join([f"{mode}:{count}" for mode, count in modes.items()])
        print(f"{disease:25} {mode_str}")
    
    # Print issues
    if issues:
        print(f"\n⚠️  POTENTIAL ISSUES ({len(issues)} found):")
        print("-" * 40)
        for issue in issues[:10]:  # Show first 10 issues
            print(f"  {issue}")
        if len(issues) > 10:
            print(f"  ... and {len(issues) - 10} more")
    else:
        print(f"\n✅ No color mode issues found!")
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    print("-" * 20)
    
    if 'RGB' in mode_counts and mode_counts['RGB'] == total_images:
        print("✅ All images are RGB - perfect for ViT!")
    elif 'RGBA' in mode_counts:
        print("⚠️  RGBA images detected - convert to RGB (handle transparency)")
    elif 'L' in mode_counts:
        print("⚠️  Grayscale images detected - convert to RGB")
    
    unusual_modes = [mode for mode in mode_counts if mode not in ['RGB', 'L', 'RGBA']]
    if unusual_modes:
        print(f"❌ Unusual modes detected: {unusual_modes} - needs investigation")
    
    print(f"\n🔧 Preprocessing needed:")
    if total_images > 0 and mode_counts.get('RGB', 0) != total_images:
        print("   image = image.convert('RGB')  # Convert all to RGB")
    else:
        print("   No color mode conversion needed!")
    
    return mode_counts, disease_modes, issues

# Sample usage
if __name__ == "__main__":
    mode_counts, disease_modes, issues = analyze_color_modes()
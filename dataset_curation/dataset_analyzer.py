#!/usr/bin/env python
# dataset_analyzer.py
# -------------------------------------------------------------
# Analyze MLB pitch dataset to understand class distribution
# -------------------------------------------------------------

import json
import os
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

def load_dataset_info(json_path='data/mlb-youtube-segmented.json'):
    """Load the dataset JSON and return basic info"""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"✅ Loaded dataset with {len(data)} total entries")
        return data
    except FileNotFoundError:
        print(f"❌ Could not find {json_path}")
        return None
    except json.JSONDecodeError as e:
        print(f"❌ Error parsing JSON: {e}")
        return None

def analyze_pitch_types(data):
    """Analyze pitch type distribution"""
    print(f"\n{'='*60}")
    print("📊 PITCH TYPE ANALYSIS")
    print(f"{'='*60}")
    
    # Count pitch types with error handling
    pitch_types = []
    missing_type_count = 0
    
    for entry_id, entry in data.items():
        if isinstance(entry, dict) and 'type' in entry:
            pitch_types.append(entry['type'])
        else:
            missing_type_count += 1
            if missing_type_count <= 5:  # Show first 5 examples
                print(f"⚠️  Entry {entry_id} missing 'type' field: {entry}")
    
    if missing_type_count > 0:
        print(f"\n⚠️  Warning: {missing_type_count} entries missing 'type' field")
    
    if not pitch_types:
        print("❌ No valid pitch types found!")
        return Counter()
    
    type_counts = Counter(pitch_types)
    
    print(f"\n🎯 Unique Pitch Types Found: {len(type_counts)}")
    print(f"📈 Total Valid Pitches: {len(pitch_types)}")
    
    # Detailed breakdown
    print(f"\n📋 Detailed Breakdown:")
    print(f"{'Pitch Type':<15} {'Count':<8} {'Percentage':<12} {'Bar'}")
    print("-" * 50)
    
    for pitch_type, count in type_counts.most_common():
        percentage = (count / len(pitch_types)) * 100
        bar = "█" * int(percentage / 2)  # Scale bar for readability
        print(f"{pitch_type:<15} {count:<8} {percentage:>6.1f}%     {bar}")
    
    return type_counts

def analyze_subsets(data):
    """Analyze training vs testing split"""
    print(f"\n{'='*60}")
    print("📂 SUBSET ANALYSIS")
    print(f"{'='*60}")
    
    # Count subsets with error handling
    subsets = []
    for entry in data.values():
        if isinstance(entry, dict) and 'subset' in entry:
            subsets.append(entry['subset'])
    
    if not subsets:
        print("❌ No subset information found!")
        return Counter(), pd.DataFrame()
    
    subset_counts = Counter(subsets)
    
    for subset, count in subset_counts.items():
        percentage = (count / len(subsets)) * 100
        print(f"{subset.title():<10}: {count:>6} samples ({percentage:>5.1f}%)")
    
    # Cross-tabulation: pitch type by subset
    print(f"\n📊 Pitch Types by Subset:")
    
    subset_pitch_types = defaultdict(lambda: defaultdict(int))
    valid_entries = []
    
    for entry in data.values():
        if isinstance(entry, dict) and 'subset' in entry and 'type' in entry:
            subset_pitch_types[entry['subset']][entry['type']] += 1
            valid_entries.append(entry)
    
    if not valid_entries:
        print("❌ No valid entries with both subset and type!")
        return subset_counts, pd.DataFrame()
    
    # Create DataFrame for better visualization
    pitch_types = set(entry['type'] for entry in valid_entries)
    subset_df = pd.DataFrame(index=sorted(pitch_types))
    
    for subset in sorted(subset_pitch_types.keys()):
        subset_df[subset] = [subset_pitch_types[subset][pitch_type] for pitch_type in sorted(pitch_types)]
    
    print(subset_df)
    
    return subset_counts, subset_df

def analyze_speeds(data):
    """Analyze pitch speed distribution"""
    print(f"\n{'='*60}")
    print("⚡ SPEED ANALYSIS")
    print(f"{'='*60}")
    
    # Group speeds by pitch type with error handling
    speed_by_type = defaultdict(list)
    valid_speed_count = 0
    missing_speed_count = 0
    
    for entry in data.values():
        if isinstance(entry, dict) and 'type' in entry:
            if 'speed' in entry and entry['speed'] is not None:
                try:
                    speed = float(entry['speed'])
                    speed_by_type[entry['type']].append(speed)
                    valid_speed_count += 1
                except (ValueError, TypeError):
                    missing_speed_count += 1
            else:
                missing_speed_count += 1
    
    print(f"📊 Speed data available for {valid_speed_count} pitches")
    if missing_speed_count > 0:
        print(f"⚠️  {missing_speed_count} pitches missing speed data")
    
    if not speed_by_type:
        print("❌ No valid speed data found!")
        return {}
    
    print(f"\n🎯 Speed Statistics by Pitch Type:")
    print(f"{'Pitch Type':<15} {'Count':<8} {'Mean':<8} {'Min':<8} {'Max':<8} {'Std':<8}")
    print("-" * 70)
    
    for pitch_type, speeds in speed_by_type.items():
        if speeds:  # Only if we have speed data
            mean_speed = np.mean(speeds)
            min_speed = np.min(speeds)
            max_speed = np.max(speeds)
            std_speed = np.std(speeds)
            print(f"{pitch_type:<15} {len(speeds):<8} {mean_speed:<8.1f} {min_speed:<8.1f} {max_speed:<8.1f} {std_speed:<8.1f}")
    
    return speed_by_type

def analyze_video_files(data, video_folders=['baseline_data', 'clips']):
    """Check which videos actually exist"""
    print(f"\n{'='*60}")
    print("📹 VIDEO FILE ANALYSIS")
    print(f"{'='*60}")
    
    available_videos = set()
    missing_videos = []
    
    # Check each folder
    for folder in video_folders:
        if os.path.exists(folder):
            files = os.listdir(folder)
            video_files = [f for f in files if f.endswith(('.mp4', '.mov', '.avi'))]
            print(f"\n📁 {folder}/: {len(video_files)} video files")
            
            # Extract video IDs (remove extension)
            folder_ids = set(os.path.splitext(f)[0] for f in video_files)
            available_videos.update(folder_ids)
        else:
            print(f"\n❌ Folder '{folder}' not found")
    
    # Check which JSON entries have corresponding videos
    json_ids = set(data.keys())
    available_in_json = available_videos & json_ids
    missing_in_json = available_videos - json_ids
    missing_videos = json_ids - available_videos
    
    print(f"\n📊 Video-JSON Matching:")
    print(f"  JSON entries: {len(json_ids)}")
    print(f"  Video files found: {len(available_videos)}")
    print(f"  ✅ Matching (usable): {len(available_in_json)}")
    print(f"  ❌ Videos without JSON: {len(missing_in_json)}")
    print(f"  ❌ JSON without videos: {len(missing_videos)}")
    
    if missing_videos:
        print(f"\n⚠️  First 10 missing videos: {list(missing_videos)[:10]}")
    
    # Analyze usable data by pitch type
    usable_data = {}
    for vid_id in available_in_json:
        entry = data[vid_id]
        if isinstance(entry, dict) and 'type' in entry:
            usable_data[vid_id] = entry
    
    usable_types = Counter(entry['type'] for entry in usable_data.values())
    
    print(f"\n🎯 Usable Data by Pitch Type:")
    print(f"{'Pitch Type':<15} {'Available':<10} {'Percentage'}")
    print("-" * 40)
    
    total_usable = len(usable_data)
    if total_usable > 0:
        for pitch_type, count in usable_types.most_common():
            percentage = (count / total_usable) * 100
            print(f"{pitch_type:<15} {count:<10} {percentage:>6.1f}%")
    else:
        print("❌ No usable data found!")
    
    return usable_data, usable_types

def create_visualizations(type_counts, speed_by_type, subset_df):
    """Create visualization plots"""
    print(f"\n{'='*60}")
    print("📈 CREATING VISUALIZATIONS")
    print(f"{'='*60}")
    
    # Set up the plotting style
    plt.style.use('default')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('MLB Pitch Dataset Analysis', fontsize=16, fontweight='bold')
    
    # 1. Pitch Type Distribution (Pie Chart)
    ax1 = axes[0, 0]
    pitch_types, counts = zip(*type_counts.most_common())
    colors = plt.cm.Set3(np.linspace(0, 1, len(pitch_types)))
    wedges, texts, autotexts = ax1.pie(counts, labels=pitch_types, autopct='%1.1f%%', 
                                       colors=colors, startangle=90)
    ax1.set_title('Pitch Type Distribution', fontweight='bold')
    
    # 2. Pitch Type Distribution (Bar Chart)
    ax2 = axes[0, 1]
    bars = ax2.bar(pitch_types, counts, color=colors)
    ax2.set_title('Pitch Type Counts', fontweight='bold')
    ax2.set_xlabel('Pitch Type')
    ax2.set_ylabel('Count')
    ax2.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{int(height)}', ha='center', va='bottom')
    
    # 3. Speed Distribution by Pitch Type
    ax3 = axes[1, 0]
    if speed_by_type:
        speed_data = [speeds for speeds in speed_by_type.values() if speeds]
        speed_labels = [pitch_type for pitch_type, speeds in speed_by_type.items() if speeds]
        
        if speed_data:
            box_plot = ax3.boxplot(speed_data, labels=speed_labels, patch_artist=True)
            ax3.set_title('Speed Distribution by Pitch Type', fontweight='bold')
            ax3.set_xlabel('Pitch Type')
            ax3.set_ylabel('Speed (mph)')
            ax3.tick_params(axis='x', rotation=45)
            
            # Color the boxes
            for patch, color in zip(box_plot['boxes'], colors[:len(speed_data)]):
                patch.set_facecolor(color)
    
    # 4. Training vs Testing Split
    ax4 = axes[1, 1]
    if not subset_df.empty:
        subset_df.plot(kind='bar', ax=ax4, color=['lightblue', 'lightcoral'])
        ax4.set_title('Training vs Testing Split by Pitch Type', fontweight='bold')
        ax4.set_xlabel('Pitch Type')
        ax4.set_ylabel('Count')
        ax4.tick_params(axis='x', rotation=45)
        ax4.legend()
    
    plt.tight_layout()
    plt.savefig('dataset_analysis.png', dpi=300, bbox_inches='tight')
    print("📊 Saved visualization to 'dataset_analysis.png'")
    plt.show()

def generate_recommendations(usable_types, total_usable):
    """Generate actionable recommendations"""
    print(f"\n{'='*60}")
    print("💡 RECOMMENDATIONS")
    print(f"{'='*60}")
    
    # Calculate statistics
    min_samples = min(usable_types.values())
    max_samples = max(usable_types.values())
    imbalance_ratio = max_samples / min_samples
    
    print(f"\n📊 Dataset Health Check:")
    print(f"  Total usable samples: {total_usable}")
    print(f"  Number of classes: {len(usable_types)}")
    print(f"  Smallest class: {min_samples} samples")
    print(f"  Largest class: {max_samples} samples")
    print(f"  Imbalance ratio: {imbalance_ratio:.1f}:1")
    
    print(f"\n🎯 Actionable Recommendations:")
    
    # 1. Class imbalance
    if imbalance_ratio > 5:
        print(f"  ⚠️  SEVERE class imbalance detected!")
        print(f"     → Use class weights in loss function")
        print(f"     → Consider combining rare classes")
        print(f"     → Use stratified sampling")
    elif imbalance_ratio > 3:
        print(f"  ⚠️  Moderate class imbalance detected")
        print(f"     → Use weighted loss function")
    else:
        print(f"  ✅ Class distribution is reasonable")
    
    # 2. Sample size recommendations
    if total_usable < 100:
        print(f"  ❌ Dataset too small for deep learning!")
        print(f"     → Minimum 500+ samples recommended")
        print(f"     → Consider data augmentation")
        print(f"     → Use pre-trained models with fine-tuning")
    elif total_usable < 500:
        print(f"  ⚠️  Small dataset - high overfitting risk")
        print(f"     → Use strong regularization (dropout 0.7+)")
        print(f"     → Freeze early layers")
        print(f"     → Aggressive data augmentation")
    else:
        print(f"  ✅ Dataset size adequate for training")
    
    # 3. Model recommendations
    print(f"\n🔧 Model Architecture Suggestions:")
    if len(usable_types) > 5:
        print(f"  → Consider reducing to 3-4 main classes")
        print(f"  → Group similar pitches (fastball family, breaking balls)")
    
    if any(count < 20 for count in usable_types.values()):
        rare_classes = [name for name, count in usable_types.items() if count < 20]
        print(f"  → Remove or combine rare classes: {rare_classes}")
    
    # 4. Training recommendations
    print(f"\n🎯 Training Strategy:")
    print(f"  → Batch size: {min(8, total_usable // 10)}")
    print(f"  → Learning rate: 1e-4 to 5e-4")
    print(f"  → Epochs: 15-30")
    print(f"  → Validation split: 20%")
    print(f"  → Use early stopping")

def main():
    """Main analysis function"""
    print("🏀 MLB Pitch Dataset Analyzer")
    print("=" * 60)
    
    # Load data
    data = load_dataset_info()
    if data is None:
        return
    
    # Run all analyses
    type_counts = analyze_pitch_types(data)
    subset_counts, subset_df = analyze_subsets(data)
    speed_by_type = analyze_speeds(data)
    usable_data, usable_types = analyze_video_files(data)
    
    # Create visualizations
    try:
        create_visualizations(type_counts, speed_by_type, subset_df)
    except Exception as e:
        print(f"⚠️  Could not create visualizations: {e}")
    
    # Generate recommendations
    generate_recommendations(usable_types, len(usable_data))
    
    print(f"\n{'='*60}")
    print("✅ Analysis Complete!")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()

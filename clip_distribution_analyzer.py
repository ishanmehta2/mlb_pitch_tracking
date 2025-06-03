#!/usr/bin/env python
# clip_distribution_analyzer.py
# -------------------------------------------------------------
# Analyze the distribution of downloaded MLB pitch clips
# -------------------------------------------------------------

import json
import os
from pathlib import Path
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def load_dataset_info(json_path='data/mlb-youtube-segmented.json'):
    """Load the dataset JSON"""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"✅ Loaded dataset JSON with {len(data)} total entries")
        return data
    except FileNotFoundError:
        print(f"❌ Could not find {json_path}")
        print("Make sure the JSON file exists and the path is correct")
        return None

def scan_downloaded_clips(clips_folder):
    """Scan the downloaded clips folder and extract video IDs"""
    clips_path = Path(clips_folder)
    
    if not clips_path.exists():
        print(f"❌ Clips folder not found: {clips_folder}")
        return []
    
    # Find all .mp4 files
    video_files = list(clips_path.glob("*.mp4"))
    print(f"📁 Found {len(video_files)} .mp4 files in {clips_folder}")
    
    # Extract video IDs (filename without extension)
    video_ids = []
    for video_file in video_files:
        video_id = video_file.stem  # filename without extension
        video_ids.append(video_id)
    
    print(f"🎯 Extracted {len(video_ids)} video IDs")
    return video_ids

def analyze_downloaded_distribution(video_ids, dataset_json):
    """Analyze the pitch type distribution of downloaded clips"""
    print(f"\n{'='*60}")
    print("📊 DOWNLOADED CLIPS ANALYSIS")
    print(f"{'='*60}")
    
    # Match video IDs with JSON data
    found_clips = {}
    missing_clips = []
    
    for video_id in video_ids:
        if video_id in dataset_json:
            clip_info = dataset_json[video_id]
            if isinstance(clip_info, dict) and 'type' in clip_info:
                found_clips[video_id] = clip_info
            else:
                missing_clips.append(video_id)
                print(f"⚠️  {video_id}: No 'type' field in JSON")
        else:
            missing_clips.append(video_id)
            print(f"⚠️  {video_id}: Not found in JSON")
    
    print(f"\n📈 Matching Results:")
    print(f"   Total downloaded clips: {len(video_ids)}")
    print(f"   ✅ Found in JSON: {len(found_clips)}")
    print(f"   ❌ Missing from JSON: {len(missing_clips)}")
    
    if missing_clips:
        print(f"\n⚠️  First 10 missing clips: {missing_clips[:10]}")
    
    return found_clips, missing_clips

def create_distribution_analysis(found_clips):
    """Create detailed distribution analysis"""
    print(f"\n{'='*60}")
    print("🎯 PITCH TYPE DISTRIBUTION")
    print(f"{'='*60}")
    
    # Count pitch types
    pitch_counts = Counter()
    speed_by_type = defaultdict(list)
    clips_by_type = defaultdict(list)
    
    for clip_id, clip_info in found_clips.items():
        pitch_type = clip_info['type']
        pitch_counts[pitch_type] += 1
        clips_by_type[pitch_type].append(clip_id)
        
        # Collect speed information
        if 'speed' in clip_info and clip_info['speed'] is not None:
            try:
                speed = float(clip_info['speed'])
                speed_by_type[pitch_type].append(speed)
            except (ValueError, TypeError):
                pass
    
    total_clips = len(found_clips)
    
    print(f"\n📋 Current Distribution ({total_clips} total clips):")
    print(f"{'Pitch Type':<15} {'Count':<8} {'Percentage':<12} {'Avg Speed':<10} {'Visual'}")
    print("-" * 70)
    
    for pitch_type, count in pitch_counts.most_common():
        percentage = (count / total_clips) * 100
        
        # Calculate average speed
        speeds = speed_by_type.get(pitch_type, [])
        avg_speed = np.mean(speeds) if speeds else 0
        speed_str = f"{avg_speed:.1f} mph" if avg_speed > 0 else "N/A"
        
        # Visual bar
        bar_length = int(percentage / 2)  # Scale for readability
        bar = "█" * bar_length
        
        print(f"{pitch_type:<15} {count:<8} {percentage:>6.1f}%     {speed_str:<10} {bar}")
    
    return pitch_counts, clips_by_type, speed_by_type

def calculate_balance_targets(pitch_counts, target_size=None):
    """Calculate how many clips of each type we need for a balanced dataset"""
    print(f"\n{'='*60}")
    print("⚖️  DATASET BALANCING ANALYSIS")
    print(f"{'='*60}")
    
    total_clips = sum(pitch_counts.values())
    num_classes = len(pitch_counts)
    
    if target_size is None:
        # Use the count of the most common class as target
        target_size = max(pitch_counts.values())
    
    print(f"🎯 Balancing Strategy:")
    print(f"   Current total clips: {total_clips}")
    print(f"   Number of classes: {num_classes}")
    print(f"   Target per class: {target_size}")
    print(f"   Balanced dataset size: {target_size * num_classes}")
    
    print(f"\n📊 Balance Analysis:")
    print(f"{'Pitch Type':<15} {'Current':<8} {'Target':<8} {'Need':<8} {'Action'}")
    print("-" * 60)
    
    balance_plan = {}
    
    for pitch_type, current_count in pitch_counts.most_common():
        needed = max(0, target_size - current_count)
        
        if needed == 0:
            action = "✅ Enough (trim excess)"
            trim_amount = current_count - target_size
            balance_plan[pitch_type] = {'action': 'trim', 'amount': trim_amount}
        elif needed <= 50:
            action = "🟡 Need a few more"
            balance_plan[pitch_type] = {'action': 'download', 'amount': needed}
        else:
            action = "🔴 Need many more"
            balance_plan[pitch_type] = {'action': 'download', 'amount': needed}
        
        print(f"{pitch_type:<15} {current_count:<8} {target_size:<8} {needed:<8} {action}")
    
    return balance_plan, target_size

def generate_clip_selection_lists(clips_by_type, balance_plan, target_size, output_dir="clip_selection"):
    """Generate lists of clips to keep for balanced dataset"""
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print("📝 GENERATING CLIP SELECTION LISTS")
    print(f"{'='*60}")
    
    selected_clips = {}
    
    for pitch_type, clips in clips_by_type.items():
        plan = balance_plan.get(pitch_type, {'action': 'keep', 'amount': 0})
        
        if plan['action'] == 'trim' or len(clips) >= target_size:
            # Randomly select clips to keep
            import random
            random.seed(42)  # For reproducibility
            selected = random.sample(clips, min(target_size, len(clips)))
        else:
            # Keep all clips (need to download more)
            selected = clips
        
        selected_clips[pitch_type] = selected
        
        # Save list to file
        filename = f"{output_dir}/{pitch_type}_selected_clips.txt"
        with open(filename, 'w') as f:
            f.write(f"# Selected {pitch_type} clips for balanced dataset\n")
            f.write(f"# Total selected: {len(selected)} clips\n\n")
            for clip_id in selected:
                f.write(f"{clip_id}.mp4\n")
        
        print(f"📝 {pitch_type}: Selected {len(selected)}/{len(clips)} clips → {filename}")
    
    # Create master list of all selected clips
    all_selected = []
    for clips in selected_clips.values():
        all_selected.extend(clips)
    
    master_file = f"{output_dir}/balanced_dataset_clips.txt"
    with open(master_file, 'w') as f:
        f.write(f"# Balanced MLB Pitch Dataset\n")
        f.write(f"# Total clips: {len(all_selected)}\n")
        f.write(f"# Target per class: {target_size}\n\n")
        
        for pitch_type, clips in selected_clips.items():
            f.write(f"\n# {pitch_type.upper()} ({len(clips)} clips)\n")
            for clip_id in clips:
                f.write(f"{clip_id}.mp4\n")
    
    print(f"\n📋 Master selection: {len(all_selected)} clips → {master_file}")
    
    return selected_clips

def create_visualizations(pitch_counts, output_dir="clip_analysis"):
    """Create visualization of current distribution"""
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print("📊 CREATING VISUALIZATIONS")
    print(f"{'='*60}")
    
    # Set up the plotting
    plt.style.use('default')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('Downloaded MLB Pitch Clips Distribution', fontsize=16, fontweight='bold')
    
    # Pie chart
    pitch_types = list(pitch_counts.keys())
    counts = list(pitch_counts.values())
    colors = plt.cm.Set3(np.linspace(0, 1, len(pitch_types)))
    
    wedges, texts, autotexts = ax1.pie(counts, labels=pitch_types, autopct='%1.1f%%', 
                                       colors=colors, startangle=90)
    ax1.set_title('Distribution (Pie Chart)', fontweight='bold')
    
    # Bar chart
    bars = ax2.bar(pitch_types, counts, color=colors)
    ax2.set_title('Distribution (Bar Chart)', fontweight='bold')
    ax2.set_xlabel('Pitch Type')
    ax2.set_ylabel('Number of Clips')
    ax2.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{int(height)}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    # Save plot
    plot_file = f"{output_dir}/clip_distribution.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"📊 Distribution plot saved: {plot_file}")
    plt.show()

def main():
    """Main analysis function"""
    print("📊 MLB Clips Distribution Analyzer")
    print("=" * 60)
    
    # Configuration
    CLIPS_FOLDER = "/Users/ishan/Downloads/231n_clips"
    JSON_PATH = "data/mlb-youtube-segmented.json"
    TARGET_CLIPS_PER_CLASS = None  # Will auto-calculate
    
    # Load JSON data
    dataset_json = load_dataset_info(JSON_PATH)
    if dataset_json is None:
        return
    
    # Scan downloaded clips
    video_ids = scan_downloaded_clips(CLIPS_FOLDER)
    if not video_ids:
        return
    
    # Analyze distribution
    found_clips, missing_clips = analyze_downloaded_distribution(video_ids, dataset_json)
    
    if not found_clips:
        print("❌ No valid clips found for analysis")
        return
    
    # Create distribution analysis
    pitch_counts, clips_by_type, speed_by_type = create_distribution_analysis(found_clips)
    
    # Calculate balance targets
    balance_plan, target_size = calculate_balance_targets(pitch_counts, TARGET_CLIPS_PER_CLASS)
    
    # Generate clip selection lists
    selected_clips = generate_clip_selection_lists(clips_by_type, balance_plan, target_size)
    
    # Create visualizations
    create_visualizations(pitch_counts)
    
    print(f"\n🎯 SUMMARY:")
    print(f"   📁 Analyzed: {len(found_clips)} downloaded clips")
    print(f"   🎯 Classes: {len(pitch_counts)} pitch types")
    print(f"   ⚖️  Target per class: {target_size} clips")
    print(f"   📝 Selection lists saved to clip_selection/")
    print(f"   📊 Visualizations saved to clip_analysis/")
    
    print(f"\n📋 Next Steps:")
    print(f"   1. Review the balance analysis above")
    print(f"   2. Check clip_selection/ for lists of clips to keep")
    print(f"   3. Download more clips for underrepresented classes")
    print(f"   4. Use balanced_dataset_clips.txt for training")

if __name__ == "__main__":
    main()
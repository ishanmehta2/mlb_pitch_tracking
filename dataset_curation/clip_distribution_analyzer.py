#!/usr/bin/env python

import json
import os
from pathlib import Path
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def load_dataset_info(json_path='data/mlb-youtube-segmented.json'):
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        return None

def scan_downloaded_clips(clips_folder):
    clips_path = Path(clips_folder)
    
    if not clips_path.exists():
        return []
    
    video_files = list(clips_path.glob("*.mp4"))
    
    video_ids = []
    for video_file in video_files:
        video_id = video_file.stem
        video_ids.append(video_id)
    
    return video_ids

def analyze_downloaded_distribution(video_ids, dataset_json):
    found_clips = {}
    missing_clips = []
    
    for video_id in video_ids:
        if video_id in dataset_json:
            clip_info = dataset_json[video_id]
            if isinstance(clip_info, dict) and 'type' in clip_info:
                found_clips[video_id] = clip_info
            else:
                missing_clips.append(video_id)
        else:
            missing_clips.append(video_id)
    
    return found_clips, missing_clips

def create_distribution_analysis(found_clips):
    pitch_counts = Counter()
    speed_by_type = defaultdict(list)
    clips_by_type = defaultdict(list)
    
    for clip_id, clip_info in found_clips.items():
        pitch_type = clip_info['type']
        pitch_counts[pitch_type] += 1
        clips_by_type[pitch_type].append(clip_id)
        
        if 'speed' in clip_info and clip_info['speed'] is not None:
            try:
                speed = float(clip_info['speed'])
                speed_by_type[pitch_type].append(speed)
            except (ValueError, TypeError):
                pass
    
    return pitch_counts, clips_by_type, speed_by_type

def calculate_balance_targets(pitch_counts, target_size=None):
    total_clips = sum(pitch_counts.values())
    num_classes = len(pitch_counts)
    
    if target_size is None:
        target_size = max(pitch_counts.values())
    
    balance_plan = {}
    
    for pitch_type, current_count in pitch_counts.most_common():
        needed = max(0, target_size - current_count)
        
        if needed == 0:
            trim_amount = current_count - target_size
            balance_plan[pitch_type] = {'action': 'trim', 'amount': trim_amount}
        elif needed <= 50:
            balance_plan[pitch_type] = {'action': 'download', 'amount': needed}
        else:
            balance_plan[pitch_type] = {'action': 'download', 'amount': needed}
    
    return balance_plan, target_size

def generate_clip_selection_lists(clips_by_type, balance_plan, target_size, output_dir="clip_selection"):
    os.makedirs(output_dir, exist_ok=True)
    
    selected_clips = {}
    
    for pitch_type, clips in clips_by_type.items():
        plan = balance_plan.get(pitch_type, {'action': 'keep', 'amount': 0})
        
        if plan['action'] == 'trim' or len(clips) >= target_size:
            import random
            random.seed(42)
            selected = random.sample(clips, min(target_size, len(clips)))
        else:
            selected = clips
        
        selected_clips[pitch_type] = selected
        
        filename = f"{output_dir}/{pitch_type}_selected_clips.txt"
        with open(filename, 'w') as f:
            f.write(f"# Selected {pitch_type} clips for balanced dataset\n")
            f.write(f"# Total selected: {len(selected)} clips\n\n")
            for clip_id in selected:
                f.write(f"{clip_id}.mp4\n")
    
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
    
    return selected_clips

def create_visualizations(pitch_counts, output_dir="clip_analysis"):
    os.makedirs(output_dir, exist_ok=True)
    
    plt.style.use('default')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('Downloaded MLB Pitch Clips Distribution', fontsize=16, fontweight='bold')
    
    pitch_types = list(pitch_counts.keys())
    counts = list(pitch_counts.values())
    colors = plt.cm.Set3(np.linspace(0, 1, len(pitch_types)))
    
    wedges, texts, autotexts = ax1.pie(counts, labels=pitch_types, autopct='%1.1f%%', 
                                       colors=colors, startangle=90)
    ax1.set_title('Distribution (Pie Chart)', fontweight='bold')
    
    bars = ax2.bar(pitch_types, counts, color=colors)
    ax2.set_title('Distribution (Bar Chart)', fontweight='bold')
    ax2.set_xlabel('Pitch Type')
    ax2.set_ylabel('Number of Clips')
    ax2.tick_params(axis='x', rotation=45)
    
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{int(height)}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    plot_file = f"{output_dir}/clip_distribution.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.show()

def main():
    CLIPS_FOLDER = "/Users/ishan/Downloads/231n_clips"
    JSON_PATH = "data/mlb-youtube-segmented.json"
    TARGET_CLIPS_PER_CLASS = None
    
    dataset_json = load_dataset_info(JSON_PATH)
    if dataset_json is None:
        return
    
    video_ids = scan_downloaded_clips(CLIPS_FOLDER)
    if not video_ids:
        return
    
    found_clips, missing_clips = analyze_downloaded_distribution(video_ids, dataset_json)
    
    if not found_clips:
        return
    
    pitch_counts, clips_by_type, speed_by_type = create_distribution_analysis(found_clips)
    
    balance_plan, target_size = calculate_balance_targets(pitch_counts, TARGET_CLIPS_PER_CLASS)
    
    selected_clips = generate_clip_selection_lists(clips_by_type, balance_plan, target_size)
    
    create_visualizations(pitch_counts)

if __name__ == "__main__":
    main()

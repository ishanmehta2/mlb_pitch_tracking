#!/usr/bin/env python

import json
import os
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

def load_dataset_info(json_path='data/mlb-youtube-segmented.json'):
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        return None
    except json.JSONDecodeError as e:
        return None

def analyze_pitch_types(data):
    pitch_types = []
    missing_type_count = 0
    
    for entry_id, entry in data.items():
        if isinstance(entry, dict) and 'type' in entry:
            pitch_types.append(entry['type'])
        else:
            missing_type_count += 1
    
    if not pitch_types:
        return Counter()
    
    type_counts = Counter(pitch_types)
    
    return type_counts

def analyze_subsets(data):
    subsets = []
    for entry in data.values():
        if isinstance(entry, dict) and 'subset' in entry:
            subsets.append(entry['subset'])
    
    if not subsets:
        return Counter(), pd.DataFrame()
    
    subset_counts = Counter(subsets)
    
    subset_pitch_types = defaultdict(lambda: defaultdict(int))
    valid_entries = []
    
    for entry in data.values():
        if isinstance(entry, dict) and 'subset' in entry and 'type' in entry:
            subset_pitch_types[entry['subset']][entry['type']] += 1
            valid_entries.append(entry)
    
    if not valid_entries:
        return subset_counts, pd.DataFrame()
    
    pitch_types = set(entry['type'] for entry in valid_entries)
    subset_df = pd.DataFrame(index=sorted(pitch_types))
    
    for subset in sorted(subset_pitch_types.keys()):
        subset_df[subset] = [subset_pitch_types[subset][pitch_type] for pitch_type in sorted(pitch_types)]
    
    return subset_counts, subset_df

def analyze_speeds(data):
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
    
    if not speed_by_type:
        return {}
    
    return speed_by_type

def analyze_video_files(data, video_folders=['baseline_data', 'clips']):
    available_videos = set()
    missing_videos = []
    
    for folder in video_folders:
        if os.path.exists(folder):
            files = os.listdir(folder)
            video_files = [f for f in files if f.endswith(('.mp4', '.mov', '.avi'))]
            
            folder_ids = set(os.path.splitext(f)[0] for f in video_files)
            available_videos.update(folder_ids)
    
    json_ids = set(data.keys())
    available_in_json = available_videos & json_ids
    missing_in_json = available_videos - json_ids
    missing_videos = json_ids - available_videos
    
    usable_data = {}
    for vid_id in available_in_json:
        entry = data[vid_id]
        if isinstance(entry, dict) and 'type' in entry:
            usable_data[vid_id] = entry
    
    usable_types = Counter(entry['type'] for entry in usable_data.values())
    
    return usable_data, usable_types

def create_visualizations(type_counts, speed_by_type, subset_df):
    plt.style.use('default')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('MLB Pitch Dataset Analysis', fontsize=16, fontweight='bold')
    
    ax1 = axes[0, 0]
    pitch_types, counts = zip(*type_counts.most_common())
    colors = plt.cm.Set3(np.linspace(0, 1, len(pitch_types)))
    wedges, texts, autotexts = ax1.pie(counts, labels=pitch_types, autopct='%1.1f%%', 
                                       colors=colors, startangle=90)
    ax1.set_title('Pitch Type Distribution', fontweight='bold')
    
    ax2 = axes[0, 1]
    bars = ax2.bar(pitch_types, counts, color=colors)
    ax2.set_title('Pitch Type Counts', fontweight='bold')
    ax2.set_xlabel('Pitch Type')
    ax2.set_ylabel('Count')
    ax2.tick_params(axis='x', rotation=45)
    
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{int(height)}', ha='center', va='bottom')
    
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
            
            for patch, color in zip(box_plot['boxes'], colors[:len(speed_data)]):
                patch.set_facecolor(color)
    
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
    plt.show()

def generate_recommendations(usable_types, total_usable):
    min_samples = min(usable_types.values())
    max_samples = max(usable_types.values())
    imbalance_ratio = max_samples / min_samples

def main():
    data = load_dataset_info()
    if data is None:
        return
    
    type_counts = analyze_pitch_types(data)
    subset_counts, subset_df = analyze_subsets(data)
    speed_by_type = analyze_speeds(data)
    usable_data, usable_types = analyze_video_files(data)
    
    try:
        create_visualizations(type_counts, speed_by_type, subset_df)
    except Exception as e:
        pass
    
    generate_recommendations(usable_types, len(usable_data))

if __name__ == "__main__":
    main()

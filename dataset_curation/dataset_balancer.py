#!/usr/bin/env python

import json
import os
from collections import Counter, defaultdict
import random

def load_dataset_info(json_path='data/mlb-youtube-segmented.json'):
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        return None

def get_existing_videos(video_folders=['baseline_data', 'clips']):
    existing_videos = set()
    
    for folder in video_folders:
        if os.path.exists(folder):
            files = os.listdir(folder)
            video_files = [f for f in files if f.endswith(('.mp4', '.mov', '.avi'))]
            folder_ids = set(os.path.splitext(f)[0] for f in video_files)
            existing_videos.update(folder_ids)
    
    return existing_videos

def analyze_current_distribution(data, existing_videos):
    current_counts = Counter()
    missing_counts = Counter()
    
    for video_id, entry in data.items():
        if isinstance(entry, dict) and 'type' in entry:
            pitch_type = entry['type']
            if video_id in existing_videos:
                current_counts[pitch_type] += 1
            else:
                missing_counts[pitch_type] += 1
    
    return current_counts, missing_counts

def find_target_balance(current_counts, target_per_class=500):
    targets = {}
    needed = {}
    
    for pitch_type, current in current_counts.items():
        targets[pitch_type] = target_per_class
        needed[pitch_type] = max(0, target_per_class - current)
    
    return needed

def find_games_to_download(data, existing_videos, needed_counts, max_games=20):
    games_data = defaultdict(lambda: {'pitches': [], 'pitch_counts': Counter(), 'available_pitches': []})
    
    for video_id, entry in data.items():
        if isinstance(entry, dict) and 'type' in entry and 'url' in entry:
            url = entry['url']
            pitch_type = entry['type']
            
            games_data[url]['pitches'].append((video_id, pitch_type))
            games_data[url]['pitch_counts'][pitch_type] += 1
            
            if video_id not in existing_videos:
                games_data[url]['available_pitches'].append((video_id, pitch_type))
    
    game_scores = []
    
    for url, game_info in games_data.items():
        score = 0
        available_counts = Counter()
        
        for video_id, pitch_type in game_info['available_pitches']:
            available_counts[pitch_type] += 1
        
        priority_breakdown = {}
        for pitch_type, needed in needed_counts.items():
            if needed > 0 and available_counts[pitch_type] > 0:
                type_score = available_counts[pitch_type] * (needed / 100)
                score += type_score
                priority_breakdown[pitch_type] = available_counts[pitch_type]
        
        if score > 0:
            game_scores.append({
                'url': url,
                'score': score,
                'total_available': len(game_info['available_pitches']),
                'priority_pitches': priority_breakdown,
                'all_available_counts': available_counts,
                'game_info': game_info
            })
    
    game_scores.sort(key=lambda x: x['score'], reverse=True)
    
    selected_games = []
    cumulative_gains = Counter()
    
    for i, game in enumerate(game_scores[:max_games]):
        selected_games.append(game)
        
        for pitch_type, count in game['priority_pitches'].items():
            cumulative_gains[pitch_type] += count
    
    return selected_games

def save_game_download_lists(selected_games, output_dir="download_lists"):
    os.makedirs(output_dir, exist_ok=True)
    
    urls_file = f"{output_dir}/priority_youtube_urls.txt"
    with open(urls_file, 'w') as f:
        f.write("# Priority YouTube URLs to download (ordered by value)\n")
        f.write("# Format: URL | Available Pitches | Priority Types\n\n")
        
        for i, game in enumerate(selected_games):
            url = game['url']
            available = game['total_available']
            priority_str = ', '.join([f"{pt}:{cnt}" for pt, cnt in game['priority_pitches'].items()])
            
            f.write(f"{url}\n")
            f.write(f"# Rank {i+1}: {available} pitches available, Priority: {priority_str}\n\n")
    
    detailed_file = f"{output_dir}/game_breakdown.txt"
    with open(detailed_file, 'w') as f:
        f.write("DETAILED GAME BREAKDOWN\n")
        f.write("=" * 50 + "\n\n")
        
        for i, game in enumerate(selected_games):
            f.write(f"RANK {i+1}: Score {game['score']:.1f}\n")
            f.write(f"URL: {game['url']}\n")
            f.write(f"Available pitches to download: {game['total_available']}\n")
            f.write(f"Priority breakdown: {dict(game['priority_pitches'])}\n")
            f.write(f"All available types: {dict(game['all_available_counts'])}\n")
            f.write("\nSpecific video IDs in this game:\n")
            
            for video_id, pitch_type in game['game_info']['available_pitches']:
                f.write(f"  {video_id} ({pitch_type})\n")
            
            f.write("\n" + "-" * 50 + "\n\n")
    
    simple_urls_file = f"{output_dir}/download_urls_only.txt"
    with open(simple_urls_file, 'w') as f:
        for game in selected_games:
            f.write(f"{game['url']}\n")
    
    return selected_games

def analyze_download_efficiency(selected_games, needed_counts):
    total_videos_to_download = sum(game['total_available'] for game in selected_games)
    total_priority_pitches = sum(sum(game['priority_pitches'].values()) for game in selected_games)
    
    cumulative_gains = Counter()
    for game in selected_games:
        for pitch_type, count in game['priority_pitches'].items():
            cumulative_gains[pitch_type] += count
    
    unsatisfied_classes = [pt for pt, need in needed_counts.items() 
                          if need > 0 and cumulative_gains.get(pt, 0) < need * 0.5]

def save_download_lists(download_lists, output_dir="download_lists"):
    pass

def generate_download_summary(download_lists, data):
    pass

def main():
    TARGET_PER_CLASS = 300
    MAX_GAMES_TO_DOWNLOAD = 15
    
    data = load_dataset_info()
    if data is None:
        return
    
    existing_videos = get_existing_videos()
    
    current_counts, missing_counts = analyze_current_distribution(data, existing_videos)
    
    needed_counts = find_target_balance(current_counts, TARGET_PER_CLASS)
    
    selected_games = find_games_to_download(data, existing_videos, needed_counts, MAX_GAMES_TO_DOWNLOAD)
    
    save_game_download_lists(selected_games, "download_lists")
    
    analyze_download_efficiency(selected_games, needed_counts)

if __name__ == "__main__":
    main()

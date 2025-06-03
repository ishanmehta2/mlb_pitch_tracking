#!/usr/bin/env python
# dataset_balancer.py
# -------------------------------------------------------------
# Find specific video IDs to download for dataset balancing
# -------------------------------------------------------------

import json
import os
from collections import Counter, defaultdict
import random

def load_dataset_info(json_path='data/mlb-youtube-segmented.json'):
    """Load the dataset JSON"""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"✅ Loaded dataset with {len(data)} total entries")
        return data
    except FileNotFoundError:
        print(f"❌ Could not find {json_path}")
        return None

def get_existing_videos(video_folders=['baseline_data', 'clips']):
    """Get set of video IDs we already have"""
    existing_videos = set()
    
    for folder in video_folders:
        if os.path.exists(folder):
            files = os.listdir(folder)
            video_files = [f for f in files if f.endswith(('.mp4', '.mov', '.avi'))]
            # Extract video IDs (remove extension)
            folder_ids = set(os.path.splitext(f)[0] for f in video_files)
            existing_videos.update(folder_ids)
            print(f"📁 Found {len(folder_ids)} videos in {folder}/")
    
    print(f"📊 Total existing videos: {len(existing_videos)}")
    return existing_videos

def analyze_current_distribution(data, existing_videos):
    """Analyze current class distribution in existing videos"""
    print(f"\n{'='*60}")
    print("📊 CURRENT DATASET ANALYSIS")
    print(f"{'='*60}")
    
    # Count what we currently have
    current_counts = Counter()
    missing_counts = Counter()
    
    for video_id, entry in data.items():
        if isinstance(entry, dict) and 'type' in entry:
            pitch_type = entry['type']
            if video_id in existing_videos:
                current_counts[pitch_type] += 1
            else:
                missing_counts[pitch_type] += 1
    
    print(f"\n🎯 Current Distribution (what you have):")
    print(f"{'Pitch Type':<15} {'Current':<8} {'Available':<10} {'Total':<8} {'Coverage'}")
    print("-" * 65)
    
    for pitch_type in sorted(set(current_counts.keys()) | set(missing_counts.keys())):
        current = current_counts[pitch_type]
        missing = missing_counts[pitch_type]
        total = current + missing
        coverage = (current / total * 100) if total > 0 else 0
        print(f"{pitch_type:<15} {current:<8} {missing:<10} {total:<8} {coverage:>6.1f}%")
    
    return current_counts, missing_counts

def find_target_balance(current_counts, target_per_class=500):
    """Determine how many more of each class we need"""
    print(f"\n{'='*60}")
    print("🎯 BALANCING STRATEGY")
    print(f"{'='*60}")
    
    targets = {}
    needed = {}
    
    print(f"\n📋 Target: {target_per_class} samples per class")
    print(f"{'Pitch Type':<15} {'Current':<8} {'Needed':<8} {'Priority'}")
    print("-" * 50)
    
    for pitch_type, current in current_counts.items():
        targets[pitch_type] = target_per_class
        needed[pitch_type] = max(0, target_per_class - current)
        
        if needed[pitch_type] == 0:
            priority = "✅ Complete"
        elif needed[pitch_type] > 200:
            priority = "🔴 HIGH"
        elif needed[pitch_type] > 100:
            priority = "🟡 MEDIUM"
        else:
            priority = "🟢 LOW"
            
        print(f"{pitch_type:<15} {current:<8} {needed[pitch_type]:<8} {priority}")
    
    return needed

def find_games_to_download(data, existing_videos, needed_counts, max_games=20):
    """Find which complete games to download based on underrepresented pitch types"""
    print(f"\n{'='*60}")
    print("🎮 GAMES TO DOWNLOAD (by YouTube URL)")
    print(f"{'='*60}")
    
    # Group all pitches by YouTube URL (game)
    games_data = defaultdict(lambda: {'pitches': [], 'pitch_counts': Counter(), 'available_pitches': []})
    
    for video_id, entry in data.items():
        if isinstance(entry, dict) and 'type' in entry and 'url' in entry:
            url = entry['url']
            pitch_type = entry['type']
            
            games_data[url]['pitches'].append((video_id, pitch_type))
            games_data[url]['pitch_counts'][pitch_type] += 1
            
            # Track which pitches are available to download (not already downloaded)
            if video_id not in existing_videos:
                games_data[url]['available_pitches'].append((video_id, pitch_type))
    
    print(f"📊 Found {len(games_data)} unique games in dataset")
    
    # Score each game based on how many underrepresented pitch types it contains
    game_scores = []
    
    for url, game_info in games_data.items():
        score = 0
        available_counts = Counter()
        
        # Count available pitches we haven't downloaded yet
        for video_id, pitch_type in game_info['available_pitches']:
            available_counts[pitch_type] += 1
        
        # Calculate priority score based on needed pitch types
        priority_breakdown = {}
        for pitch_type, needed in needed_counts.items():
            if needed > 0 and available_counts[pitch_type] > 0:
                # Higher score for more needed pitch types
                type_score = available_counts[pitch_type] * (needed / 100)  # Normalize by need
                score += type_score
                priority_breakdown[pitch_type] = available_counts[pitch_type]
        
        if score > 0:  # Only include games with needed pitch types
            game_scores.append({
                'url': url,
                'score': score,
                'total_available': len(game_info['available_pitches']),
                'priority_pitches': priority_breakdown,
                'all_available_counts': available_counts,
                'game_info': game_info
            })
    
    # Sort by score (highest first)
    game_scores.sort(key=lambda x: x['score'], reverse=True)
    
    # Display top games
    print(f"\n🏆 TOP {min(max_games, len(game_scores))} PRIORITY GAMES TO DOWNLOAD:")
    print(f"{'Rank':<4} {'Score':<8} {'Available':<10} {'Priority Pitch Types'}")
    print("-" * 80)
    
    selected_games = []
    cumulative_gains = Counter()
    
    for i, game in enumerate(game_scores[:max_games]):
        rank = i + 1
        score = game['score']
        available = game['total_available']
        
        # Format priority pitch types
        priority_str = ', '.join([f"{pt}:{cnt}" for pt, cnt in game['priority_pitches'].items()])
        
        print(f"{rank:<4} {score:<8.1f} {available:<10} {priority_str}")
        
        selected_games.append(game)
        
        # Track cumulative gains
        for pitch_type, count in game['priority_pitches'].items():
            cumulative_gains[pitch_type] += count
    
    print(f"\n📈 CUMULATIVE IMPACT if you download these {len(selected_games)} games:")
    for pitch_type, gain in cumulative_gains.most_common():
        current = needed_counts.get(pitch_type, 0)
        percentage = (gain / current * 100) if current > 0 else 0
        print(f"   {pitch_type}: +{gain} pitches ({percentage:.1f}% of need)")
    
    return selected_games

def save_game_download_lists(selected_games, output_dir="download_lists"):
    """Save game-based download information"""
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n💾 SAVING GAME-BASED DOWNLOAD LISTS")
    print(f"{'='*60}")
    
    # Save YouTube URLs to download
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
    
    print(f"📝 YouTube URLs: {len(selected_games)} games → {urls_file}")
    
    # Save detailed breakdown
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
    
    print(f"📊 Detailed breakdown: → {detailed_file}")
    
    # Save just the URLs for easy downloading
    simple_urls_file = f"{output_dir}/download_urls_only.txt"
    with open(simple_urls_file, 'w') as f:
        for game in selected_games:
            f.write(f"{game['url']}\n")
    
    print(f"🔗 Simple URL list: → {simple_urls_file}")
    
    return selected_games

def analyze_download_efficiency(selected_games, needed_counts):
    """Analyze how efficiently these games will balance the dataset"""
    print(f"\n{'='*60}")
    print("📊 DOWNLOAD EFFICIENCY ANALYSIS")
    print(f"{'='*60}")
    
    total_videos_to_download = sum(game['total_available'] for game in selected_games)
    total_priority_pitches = sum(sum(game['priority_pitches'].values()) for game in selected_games)
    
    print(f"\n🎯 Overall Statistics:")
    print(f"   Games to download: {len(selected_games)}")
    print(f"   Total video clips: {total_videos_to_download}")
    print(f"   Priority pitch clips: {total_priority_pitches}")
    print(f"   Efficiency: {total_priority_pitches/total_videos_to_download*100:.1f}% priority pitches")
    
    # Show impact on each needed class
    print(f"\n📈 Impact on Class Balance:")
    print(f"{'Pitch Type':<15} {'Currently Need':<15} {'Will Gain':<10} {'% Satisfied':<12} {'Remaining'}")
    print("-" * 75)
    
    cumulative_gains = Counter()
    for game in selected_games:
        for pitch_type, count in game['priority_pitches'].items():
            cumulative_gains[pitch_type] += count
    
    for pitch_type, needed in needed_counts.items():
        if needed > 0:
            gain = cumulative_gains.get(pitch_type, 0)
            satisfaction = (gain / needed * 100) if needed > 0 else 0
            remaining = max(0, needed - gain)
            
            print(f"{pitch_type:<15} {needed:<15} {gain:<10} {satisfaction:<11.1f}% {remaining}")
    
    # Recommendations
    print(f"\n💡 Recommendations:")
    unsatisfied_classes = [pt for pt, need in needed_counts.items() 
                          if need > 0 and cumulative_gains.get(pt, 0) < need * 0.5]
    
    if unsatisfied_classes:
        print(f"   ⚠️  Still need more: {', '.join(unsatisfied_classes)}")
        print(f"   → Consider downloading more games or adjusting targets")
    else:
        print(f"   ✅ This download batch will significantly improve balance!")
        print(f"   → Good distribution across all needed pitch types")

def save_download_lists(download_lists, output_dir="download_lists"):
    """Save download lists to files"""
    # This function is now replaced by save_game_download_lists
    # Keeping for backward compatibility but not used in main flow
    pass

def generate_download_summary(download_lists, data):
    """Generate a summary with additional info about videos to download"""
    # This function is now replaced by analyze_download_efficiency
    # Keeping for backward compatibility but not used in main flow
    pass

def main():
    """Main function"""
    print("⚖️  MLB Dataset Balancer - Game-Based Download Strategy")
    print("=" * 60)
    
    # Configuration
    TARGET_PER_CLASS = 300  # Adjust this based on your needs
    MAX_GAMES_TO_DOWNLOAD = 15  # Limit number of full games to download
    
    # Load data
    data = load_dataset_info()
    if data is None:
        return
    
    # Get existing videos
    existing_videos = get_existing_videos()
    
    # Analyze current distribution
    current_counts, missing_counts = analyze_current_distribution(data, existing_videos)
    
    # Determine what we need
    needed_counts = find_target_balance(current_counts, TARGET_PER_CLASS)
    
    # Find games to download (instead of individual videos)
    selected_games = find_games_to_download(data, existing_videos, needed_counts, MAX_GAMES_TO_DOWNLOAD)
    
    # Save game-based lists
    save_game_download_lists(selected_games, "download_lists")
    
    # Analyze efficiency
    analyze_download_efficiency(selected_games, needed_counts)
    
    print(f"\n🎯 QUICK ACTION ITEMS:")
    print(f"   1. Check download_lists/priority_youtube_urls.txt for games to download")
    print(f"   2. Download {len(selected_games)} complete YouTube videos")
    print(f"   3. Extract all pitch clips from each downloaded game")
    print(f"   4. This will efficiently balance your dataset!")
    
    # Show top 3 games for immediate action
    if selected_games:
        print(f"\n🏆 TOP 3 PRIORITY GAMES:")
        for i, game in enumerate(selected_games[:5]):
            priority_pitches = sum(game['priority_pitches'].values())
            total_available = game['total_available']
            efficiency = priority_pitches / total_available * 100
            
            print(f"\n   {i+1}. {game['url']}")
            print(f"      📊 {total_available} total clips, {priority_pitches} priority ({efficiency:.1f}% efficiency)")
            print(f"      🎯 Priority types: {dict(game['priority_pitches'])}")

if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()

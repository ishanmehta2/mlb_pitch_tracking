import torch
import torch.utils.data as tud
from torchvision.io import read_video
import numpy as np
from pathlib import Path
import json
import pickle
import random
from typing import Dict, List, Tuple, Optional

MEAN = torch.tensor([0.485, 0.456, 0.406])
STD = torch.tensor([0.229, 0.224, 0.225])

class SimplePitchDataset(tud.Dataset):
    
    def __init__(self, 
                 video_dir: str = 'no_contact_pitches',
                 features_dir: str = 'ball_tracking_features',
                 json_path: str = 'data/mlb-youtube-segmented.json',
                 num_frames: int = 16,
                 resolution: int = 112):
        
        self.video_dir = Path(video_dir)
        self.features_dir = Path(features_dir)
        self.num_frames = num_frames
        self.resolution = resolution
        
        self.label2idx = {'fastball': 0, 'breaking': 1, 'offspeed': 2}
        self.idx2label = {v: k for k, v in self.label2idx.items()}
        
        with open(json_path, 'r') as f:
            self.pitch_data = json.load(f)
        
        self.samples = []
        self._build_dataset()
        
        print(f"Dataset ready with {len(self.samples)} samples")
        self._print_stats()
    
    def _group_pitch_type(self, pitch_type: str) -> str:
        pitch_type = pitch_type.lower()
        if any(t in pitch_type for t in ['fastball', 'sinker', 'cutter']):
            return 'fastball'
        elif any(t in pitch_type for t in ['curve', 'slider']):
            return 'breaking'
        else:
            return 'offspeed'
    
    def _build_dataset(self):
        feature_files = list(self.features_dir.glob("*_features.pkl"))
        print(f"Found {len(feature_files)} feature files")
        
        valid_count = 0
        missing_video = 0
        missing_metadata = 0
        invalid_features = 0
        
        for feature_file in feature_files:
            video_id = feature_file.stem.replace('_features', '')
            video_path = self.video_dir / f"{video_id}.mp4"
            
            if not video_path.exists():
                missing_video += 1
                continue
            
            if video_id not in self.pitch_data:
                missing_metadata += 1
                continue
            
            try:
                with open(feature_file, 'rb') as f:
                    feature_data = pickle.load(f)
                
                if not all(key in feature_data for key in ['features', 'heatmap']):
                    print(f"Invalid feature file (missing data): {video_id}")
                    invalid_features += 1
                    continue
                
                features = feature_data['features']
                heatmap = feature_data['heatmap']
                
                if len(features) != 18:
                    print(f"Invalid feature dimensions ({len(features)}): {video_id}")
                    invalid_features += 1
                    continue
                
                if heatmap.shape != (64, 64):
                    print(f"Invalid heatmap shape {heatmap.shape}: {video_id}")
                    invalid_features += 1
                    continue
                
            except Exception as e:
                print(f"Could not load feature file for {video_id}: {e}")
                invalid_features += 1
                continue
        
            pitch_type = self.pitch_data[video_id]['type']
            grouped_type = self._group_pitch_type(pitch_type)
            
            self.samples.append({
                'video_id': video_id,
                'video_path': str(video_path),
                'feature_path': str(feature_file),
                'label': grouped_type
            })
            valid_count += 1
        
        print(f"\nDataset construction summary:")
        print(f"Valid samples: {valid_count}")
        print(f"Missing videos: {missing_video}")
        print(f"Missing metadata: {missing_metadata}")
        print(f"Invalid features: {invalid_features}")
        
        if valid_count == 0:
            raise ValueError("No valid samples found! Please check your data.")
        
        random.shuffle(self.samples)
    
    def _print_stats(self):
        class_counts = {label: 0 for label in self.label2idx.keys()}
        for sample in self.samples:
            class_counts[sample['label']] += 1
        
        print("\nClass distribution:")
        for label, count in class_counts.items():
            pct = count / len(self.samples) * 100 if self.samples else 0
            print(f"  {label:>10}: {count:4d} ({pct:5.1f}%)")
    
    def _load_video(self, video_path: str) -> Optional[torch.Tensor]:
        try:
            frames, _, _ = read_video(video_path, pts_unit="sec")
            if frames.shape[0] == 0:
                return None
            
            total_frames = frames.shape[0]
            if total_frames <= self.num_frames:
                indices = list(range(total_frames))
                indices += [total_frames - 1] * (self.num_frames - total_frames)
            else:
                indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
            
            frames = frames[indices]
        
            frames = frames.float() / 255.0
            frames = frames.permute(0, 3, 1, 2)
            
            frames = torch.nn.functional.interpolate(
                frames, size=self.resolution, mode='bilinear', align_corners=False
            )
            
            frames = (frames - MEAN[None, :, None, None]) / STD[None, :, None, None]
            
            frames = frames.permute(1, 0, 2, 3)
            
            return frames
            
        except Exception as e:
            print(f"Error loading video {video_path}: {e}")
            return None
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        video = self._load_video(sample['video_path'])
        if video is None:
            print(f"Failed to load video: {sample['video_id']}")
            return self.__getitem__((idx + 1) % len(self.samples))
        
        try:
            with open(sample['feature_path'], 'rb') as f:
                feature_data = pickle.load(f)
            
            features = torch.tensor(feature_data['features'], dtype=torch.float32)
            heatmap = torch.tensor(feature_data['heatmap'], dtype=torch.float32)
            
            if features.shape[0] != 18:
                print(f"Invalid feature dimensions for {sample['video_id']}: {features.shape}")
                return self.__getitem__((idx + 1) % len(self.samples))
            
            if heatmap.shape != (64, 64):
                print(f"Invalid heatmap shape for {sample['video_id']}: {heatmap.shape}")
                return self.__getitem__((idx + 1) % len(self.samples))
            
        except Exception as e:
            print(f"Error loading features for {sample['video_id']}: {e}")
            return self.__getitem__((idx + 1) % len(self.samples))
        
        label = torch.tensor(self.label2idx[sample['label']], dtype=torch.long)
        
        return video, features, heatmap, label
    
    def verify_dataset(self):
        print("\nVerifying dataset integrity...")
        failed_samples = []
        
        for i, sample in enumerate(self.samples):
            try:
                if not Path(sample['video_path']).exists():
                    failed_samples.append((sample['video_id'], 'video missing'))
                    continue
                
                with open(sample['feature_path'], 'rb') as f:
                    feature_data = pickle.load(f)
                
                if 'features' not in feature_data or 'heatmap' not in feature_data:
                    failed_samples.append((sample['video_id'], 'incomplete features'))
                    continue
                
                if len(feature_data['features']) != 18:
                    failed_samples.append((sample['video_id'], 'wrong feature dim'))
                    continue
                
                if feature_data['heatmap'].shape != (64, 64):
                    failed_samples.append((sample['video_id'], 'wrong heatmap shape'))
                    continue
                    
            except Exception as e:
                failed_samples.append((sample['video_id'], f'error: {str(e)}'))
        
        if failed_samples:
            print(f"  ⚠️ Found {len(failed_samples)} problematic samples:")
            for video_id, reason in failed_samples[:5]:  # Show first 5
                print(f"    - {video_id}: {reason}")
            if len(failed_samples) > 5:
                print(f"    ... and {len(failed_samples) - 5} more")
        else:
            print(f"All {len(self.samples)} samples verified successfully!")
        
        return len(failed_samples) == 0


def create_simple_data_loaders(
    train_ratio: float = 0.8,
    batch_size: int = 8,
    num_workers: int = 4,
    **dataset_kwargs
) -> Tuple[tud.DataLoader, tud.DataLoader, SimplePitchDataset]:
    
    dataset = SimplePitchDataset(**dataset_kwargs)
    
    n = len(dataset)
    n_train = int(n * train_ratio)
    
    indices = list(range(n))
    random.shuffle(indices)
    
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]
    
    train_dataset = tud.Subset(dataset, train_indices)
    val_dataset = tud.Subset(dataset, val_indices)
    
    print(f"\nData split:")
    print(f"  Train: {len(train_dataset)}")
    print(f"  Val: {len(val_dataset)}")
    
    # Create loaders
    train_loader = tud.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = tud.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, dataset


if __name__ == "__main__":
    print("Testing dataset...")
    
    dataset = SimplePitchDataset()
    
    dataset.verify_dataset()
    
    if len(dataset) > 0:
        print(f"\nLoading sample 0...")
        video, features, heatmap, label = dataset[0]
        print(f"  Video shape: {video.shape}")
        print(f"  Features shape: {features.shape}")
        print(f"  Heatmap shape: {heatmap.shape}")
        print(f"  Label: {label.item()} ({dataset.idx2label[label.item()]})")
        
        print(f"\nTesting data loader creation...")
        train_loader, val_loader, _ = create_simple_data_loaders(batch_size=4)
        print(f"Data loaders created successfully!")
        print(f"  Train batches: {len(train_loader)}")
        print(f"  Val batches: {len(val_loader)}")
        
        print(f"\nTesting batch loading...")
        for batch_idx, (videos, features, heatmaps, labels) in enumerate(train_loader):
            print(f"  Batch {batch_idx}: videos={videos.shape}, features={features.shape}")
            if batch_idx >= 2:
                break
    else:
        print("No valid samples found in dataset!")
#!/usr/bin/env python

import argparse, csv, math, os, random, time, warnings
from pathlib import Path

import torch, torch.nn as nn
import torch.utils.data as tud
import torchvision
from torchvision.io import read_video
import torch.nn.functional as F
import os
import json

try:
    from ball_tracking import extract_trajectory_features_from_video, normalize_trajectory_features, TrajectoryNet
    BALL_TRACKING_AVAILABLE = True
except ImportError:
    BALL_TRACKING_AVAILABLE = False

MEAN = torch.tensor([0.485, 0.456, 0.406])
STD  = torch.tensor([0.229, 0.224, 0.225])

class PitchVideoDataset(tud.Dataset):
    def __init__(self, num_frames: int = 16, resolution: int = 112):
        self.samples = []
        self.label2idx = {}
        self.num_frames, self.res = num_frames, resolution

        json_path = 'data/mlb-youtube-segmented.json' 

        with open(json_path, 'r', encoding='utf-8') as f:
            pitch_data = json.load(f)

        folder_path = 'baseline_data'

        for filename in os.listdir(folder_path):
            try:
                file_path = os.path.join(folder_path, filename)
                pitch_id = str(filename).split('.')[0]
                pitch_type = pitch_data[pitch_id]['type']
                self.samples.append([file_path, pitch_type])
                if pitch_type not in self.label2idx:
                        self.label2idx[pitch_type] = len(self.label2idx)
            except Exception as err:
                pass
            
        folder_path = 'clips'

        for filename in os.listdir(folder_path):
            try:
                if len(self.samples) >= 60:
                    break
                file_path = os.path.join(folder_path, filename)
                pitch_id = str(filename).split('.')[0]
                pitch_type = pitch_data[pitch_id]['type']
                self.samples.append([file_path, pitch_type])
                if pitch_type not in self.label2idx:
                        self.label2idx[pitch_type] = len(self.label2idx)
            except Exception as err:
                pass

    def __len__(self): return len(self.samples)

    def _sample_indices(self, num_total: int) -> list[int]:
        if num_total <= self.num_frames:
            return list(range(num_total)) + [num_total - 1] * (self.num_frames - num_total)
        step = num_total / self.num_frames
        return [math.floor(i * step) for i in range(self.num_frames)]

    @torch.no_grad()
    def _preprocess(self, vid: torch.Tensor) -> torch.Tensor:
        vid = vid.permute(0, 3, 1, 2).float() / 255.0
        vid = torch.nn.functional.interpolate(
            vid, size=self.res, mode="bilinear", align_corners=False
        )
        vid = (vid - MEAN[:, None, None]) / STD[:, None, None]
        return vid.permute(1, 0, 2, 3)

    def __getitem__(self, idx: int):
        path, label_str = self.samples[idx]
        frames, _, _ = read_video(path, pts_unit="sec")
        sel  = self._sample_indices(frames.shape[0])
        clip = self._preprocess(frames[sel])

        key_frame_indices = [0, len(sel)//2, len(sel)-1]
        key_frames = frames[sel][key_frame_indices]
        key_frames = self._preprocess_spatial(key_frames)

        if BALL_TRACKING_AVAILABLE:
            try:
                trajectory_features = extract_trajectory_features_from_video(path)
                trajectory_features = normalize_trajectory_features(trajectory_features)
            except Exception as e:
                trajectory_features = torch.zeros(12, dtype=torch.float32)
        else:
            trajectory_features = torch.zeros(12, dtype=torch.float32)

        label_idx = torch.tensor(self.label2idx[label_str], dtype=torch.long)
        return clip, key_frames, trajectory_features, label_idx

    @torch.no_grad()
    def _preprocess_spatial(self, frames: torch.Tensor) -> torch.Tensor:
        frames = frames.permute(0, 3, 1, 2).float() / 255.0
        frames = torch.nn.functional.interpolate(
            frames, size=224, mode="bilinear", align_corners=False
        )
        frames = (frames - MEAN[:, None, None]) / STD[:, None, None]
        return frames

class LateFusionPitchClassifier(nn.Module):
    def __init__(self, num_classes: int, dropout_rate: float = 0.5):
        super().__init__()
        
        self.temporal_branch = torchvision.models.video.r3d_18(pretrained=True)
        temporal_feat_dim = self.temporal_branch.fc.in_features
        self.temporal_branch.fc = nn.Identity()
        
        self.spatial_branch = torchvision.models.resnet18(pretrained=True)
        spatial_feat_dim = self.spatial_branch.fc.in_features
        self.spatial_branch.fc = nn.Identity()
        
        if BALL_TRACKING_AVAILABLE:
            self.trajectory_branch = TrajectoryNet(input_dim=12, hidden_dim=64, output_dim=32)
        else:
            self.trajectory_branch = nn.Sequential(
                nn.Linear(12, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU()
            )
        trajectory_feat_dim = 32
        
        self.spatial_pool = nn.AdaptiveAvgPool1d(1)
        
        self.temporal_proj = nn.Sequential(
            nn.Linear(temporal_feat_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        self.spatial_proj = nn.Sequential(
            nn.Linear(spatial_feat_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        self.trajectory_proj = nn.Sequential(
            nn.Linear(trajectory_feat_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        self.fusion_layer = nn.Sequential(
            nn.Linear(640, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes)
        )
        
        self.branch_weights = nn.Parameter(torch.tensor([1.0, 1.0, 1.0]))
        
    def forward(self, video_clip, key_frames, trajectory_features):
        batch_size = video_clip.size(0)
        
        temporal_features = self.temporal_branch(video_clip)
        temporal_features = self.temporal_proj(temporal_features)
        
        key_frames_flat = key_frames.view(-1, *key_frames.shape[2:])
        spatial_features = self.spatial_branch(key_frames_flat)
        spatial_features = spatial_features.view(batch_size, key_frames.size(1), -1)
        
        spatial_features = spatial_features.transpose(1, 2)
        spatial_features = self.spatial_pool(spatial_features).squeeze(-1)
        spatial_features = self.spatial_proj(spatial_features)
        
        trajectory_features = self.trajectory_branch(trajectory_features)
        trajectory_features = self.trajectory_proj(trajectory_features)
        
        weights = F.softmax(self.branch_weights, dim=0)
        temporal_weighted = temporal_features * weights[0]
        spatial_weighted = spatial_features * weights[1]
        trajectory_weighted = trajectory_features * weights[2]
        
        fused_features = torch.cat([temporal_weighted, spatial_weighted, trajectory_weighted], dim=1)
        
        output = self.fusion_layer(fused_features)
        
        return output

def split_dataset(ds, val_ratio=0.1, seed=42):
    n = len(ds)
    idxs = list(range(n))
    random.Random(seed).shuffle(idxs)
    val_sz = int(n * val_ratio)
    return tud.Subset(ds, idxs[val_sz:]), tud.Subset(ds, idxs[:val_sz])

def train_one_epoch(model, loader, criterion, optim, scaler, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    
    for batch_idx, (clips, key_frames, trajectory_features, labels) in enumerate(loader):
        clips = clips.to(device, non_blocking=True)
        key_frames = key_frames.to(device, non_blocking=True)
        trajectory_features = trajectory_features.to(device, non_blocking=True)
        labels = torch.as_tensor(labels, device=device)

        with torch.cuda.amp.autocast():
            out = model(clips, key_frames, trajectory_features)
            loss = criterion(out, labels)

        scaler.scale(loss).backward()
        scaler.step(optim)
        scaler.update()
        optim.zero_grad(set_to_none=True)

        running_loss += loss.item() * labels.size(0)
        _, pred = out.max(1)
        correct += pred.eq(labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    loss, correct, total = 0.0, 0, 0
    
    for clips, key_frames, trajectory_features, labels in loader:
        clips = clips.to(device)
        key_frames = key_frames.to(device)
        trajectory_features = trajectory_features.to(device)
        labels = labels.to(device)
        
        out = model(clips, key_frames, trajectory_features)
        loss += criterion(out, labels).item() * labels.size(0)
        _, pred = out.max(1)
        correct += pred.eq(labels).sum().item()
        total += labels.size(0)
        
    return loss / total, correct / total

def main(args):
    warnings.filterwarnings("ignore", category=UserWarning)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    full_ds = PitchVideoDataset(num_frames=args.frames, resolution=args.res)
    train_ds, val_ds = split_dataset(full_ds, val_ratio=0.1, seed=2025)
    
    train_loader = tud.DataLoader(train_ds, batch_size=args.bsz, shuffle=True,
                                  num_workers=0, pin_memory=True)
    val_loader   = tud.DataLoader(val_ds, batch_size=args.bsz, shuffle=False,
                                  num_workers=0, pin_memory=True)

    num_classes = len(full_ds.label2idx)

    model = LateFusionPitchClassifier(num_classes=num_classes, dropout_rate=args.dropout)
    model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    criterion = nn.CrossEntropyLoss()
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler()
    
    scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=5, gamma=0.5)

    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optim, scaler, device)
        va_loss, va_acc = evaluate(model, val_loader, criterion, device)
        dt = time.time() - t0
        
        weights = F.softmax(model.branch_weights, dim=0)

        scheduler.step()

        if va_acc > best_acc:
            best_acc = va_acc
            ckpt = {"epoch": epoch,
                    "state_dict": model.state_dict(),
                    "label2idx": full_ds.label2idx,
                    "args": vars(args)}
            torch.save(ckpt, args.out / "best_late_fusion.pt")

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Late Fusion Model for MLB Pitch Classification")
    p.add_argument("--out",     type=Path, default=Path("./checkpoints"))
    p.add_argument("--epochs",  type=int, default=15)
    p.add_argument("--bsz",     type=int, default=2)
    p.add_argument("--lr",      type=float, default=1e-4)
    p.add_argument("--frames",  type=int, default=16)
    p.add_argument("--res",     type=int, default=112)
    p.add_argument("--dropout", type=float, default=0.5)
    p.add_argument("--workers", type=int, default=min(4, os.cpu_count()//2))
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    main(args)

#!/usr/bin/env python

import argparse, math, os, random, time, warnings
from pathlib import Path
from collections import defaultdict
import pickle

import torch, torch.nn as nn
import torch.utils.data as tud
import torchvision
from torchvision.io import read_video
import torch.nn.functional as F
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score

MEAN = torch.tensor([0.485, 0.456, 0.406])
STD  = torch.tensor([0.229, 0.224, 0.225])

class SimplePitchDataset(tud.Dataset):
    def __init__(self, num_frames: int = 16, resolution: int = 112):
        self.num_frames, self.res = num_frames, resolution
        
        self.label2idx = {'fastball': 0, 'breaking': 1, 'offspeed': 2}
        
        json_path = 'data/mlb-youtube-segmented.json' 
        with open(json_path, 'r', encoding='utf-8') as f:
            pitch_data = json.load(f)

        def group_pitch_type(pitch_type):
            pitch_type = pitch_type.lower()
            if 'fastball' in pitch_type or 'sinker' in pitch_type or 'cutter' in pitch_type:
                return 'fastball'
            elif 'curve' in pitch_type or 'slider' in pitch_type:
                return 'breaking'
            else:
                return 'offspeed'

        self.samples = []
        video_dir = 'no_contact_pitches'
        
        if not os.path.exists(video_dir):
            return
        
        video_files = [f for f in os.listdir(video_dir) if f.endswith('.mp4')]
        
        for video_file in video_files:
            video_id = video_file.split('.')[0]
            video_path = os.path.join(video_dir, video_file)
            
            if video_id in pitch_data:
                pitch_type = pitch_data[video_id]['type']
                grouped_type = group_pitch_type(pitch_type)
                self.samples.append([video_path, grouped_type, video_id])

        random.shuffle(self.samples)

    def __len__(self): 
        return len(self.samples)

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
        path, label_str, video_id = self.samples[idx]
        
        try:
            frames, _, _ = read_video(path, pts_unit="sec")
            
            if frames.shape[0] == 0:
                return self.__getitem__((idx + 1) % len(self.samples))
                
            sel = self._sample_indices(frames.shape[0])
            clip = self._preprocess(frames[sel])
            
            label_idx = torch.tensor(self.label2idx[label_str], dtype=torch.long)
            return clip, label_idx
            
        except Exception as e:
            return self.__getitem__((idx + 1) % len(self.samples))

class Simple3DCNNClassifier(nn.Module):
    def __init__(self, num_classes: int, dropout_rate: float = 0.5):
        super().__init__()
        
        self.backbone = torchvision.models.video.r3d_18(pretrained=True)
        
        feature_dim = self.backbone.fc.in_features
        
        self.backbone.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, video_clip):
        return self.backbone(video_clip)

def split_dataset(ds, val_ratio=0.2, seed=42):
    n = len(ds)
    idxs = list(range(n))
    random.Random(seed).shuffle(idxs)
    val_sz = int(n * val_ratio)
    return tud.Subset(ds, idxs[val_sz:]), tud.Subset(ds, idxs[:val_sz])

def train_one_epoch(model, loader, criterion, optim, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    
    for batch_idx, (clips, labels) in enumerate(loader):
        clips = clips.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optim.zero_grad()
        out = model(clips)
        loss = criterion(out, labels)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optim.step()

        running_loss += loss.item() * labels.size(0)
        _, pred = out.max(1)
        correct += pred.eq(labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total

@torch.no_grad()
def evaluate(model, loader, criterion, device, label2idx=None, detailed=False):
    model.eval()
    loss, correct, total = 0.0, 0, 0
    
    all_preds, all_labels = [], []
    
    for clips, labels in loader:
        clips = clips.to(device)
        labels = labels.to(device)
        
        out = model(clips)
        loss += criterion(out, labels).item() * labels.size(0)
        
        _, pred = out.max(1)
        correct += pred.eq(labels).sum().item()
        total += labels.size(0)
        
        if detailed:
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    if detailed and label2idx is not None:
        print_detailed_results(all_labels, all_preds, label2idx)
        
    return loss / total, correct / total

def print_detailed_results(true_labels, pred_labels, label2idx):
    idx2label = {v: k for k, v in label2idx.items()}
    class_names = [idx2label[i] for i in range(len(label2idx))]
    
    true_labels = np.array(true_labels)
    pred_labels = np.array(pred_labels)
    
    accuracy = np.mean(true_labels == pred_labels)
    f1_macro = f1_score(true_labels, pred_labels, average='macro')
    f1_weighted = f1_score(true_labels, pred_labels, average='weighted')
    precision_macro = precision_score(true_labels, pred_labels, average='macro')
    recall_macro = recall_score(true_labels, pred_labels, average='macro')
    
    report = classification_report(true_labels, pred_labels, target_names=class_names, digits=3)
    
    cm = confusion_matrix(true_labels, pred_labels)

def main(args):
    warnings.filterwarnings("ignore", category=UserWarning)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    full_ds = SimplePitchDataset(
        num_frames=args.frames, 
        resolution=args.res
    )
    
    if len(full_ds) < 10:
        return
        
    train_ds, val_ds = split_dataset(full_ds, val_ratio=0.2, seed=2025)
    
    train_loader = tud.DataLoader(train_ds, batch_size=args.bsz, shuffle=True,
                                  num_workers=4, pin_memory=True)
    val_loader   = tud.DataLoader(val_ds, batch_size=args.bsz, shuffle=False,
                                  num_workers=4, pin_memory=True)

    num_classes = len(full_ds.label2idx)

    model = Simple3DCNNClassifier(
        num_classes=num_classes, 
        dropout_rate=args.dropout
    )
    model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    class_counts = defaultdict(int)
    for _, label_str, _ in full_ds.samples:
        class_counts[label_str] += 1
    
    total_samples = len(full_ds.samples)
    weights = []
    for i in range(num_classes):
        class_name = [k for k, v in full_ds.label2idx.items() if v == i][0]
        count = class_counts[class_name]
        
        weight = total_samples / (num_classes * count) if count > 0 else 1.0
        weights.append(weight)
    
    class_weights = torch.tensor(weights, dtype=torch.float).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    optim = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optim, mode='max', factor=0.5, patience=3, verbose=True
    )

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    best_acc = 0.0
    patience_counter = 0
    max_patience = 5
    
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optim, device)
        va_loss, va_acc = evaluate(model, val_loader, criterion, device)
        dt = time.time() - t0
        
        train_losses.append(tr_loss)
        val_losses.append(va_loss)
        train_accs.append(tr_acc)
        val_accs.append(va_acc)
        
        scheduler.step(va_acc)
        current_lr = optim.param_groups[0]['lr']
        
        if va_acc > best_acc:
            best_acc = va_acc
            patience_counter = 0
            
            ckpt = {
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "optimizer": optim.state_dict(),
                "scheduler": scheduler.state_dict(),
                "label2idx": full_ds.label2idx,
                "args": vars(args),
                "best_accuracy": best_acc
            }
            torch.save(ckpt, args.out / "best_simple_3d_cnn.pt")
        else:
            patience_counter += 1
        
        if epoch % 5 == 0:
            evaluate(model, val_loader, criterion, device, full_ds.label2idx, detailed=True)
        
        if patience_counter >= max_patience:
            break

    best_ckpt = torch.load(args.out / "best_simple_3d_cnn.pt", weights_only=False)
    model.load_state_dict(best_ckpt["state_dict"])
    
    final_results = evaluate(model, val_loader, criterion, device, full_ds.label2idx, detailed=True)

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Simple 3D CNN Pitch Classification")
    p.add_argument("--out",                type=Path, default=Path("./simple_checkpoints"))
    p.add_argument("--epochs",             type=int, default=15)
    p.add_argument("--bsz",                type=int, default=8)
    p.add_argument("--lr",                 type=float, default=1e-3)
    p.add_argument("--frames",             type=int, default=16)
    p.add_argument("--res",                type=int, default=112)
    p.add_argument("--dropout",            type=float, default=0.5)
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    main(args)

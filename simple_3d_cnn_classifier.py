#!/usr/bin/env python
# simple_3d_cnn_classifier.py
# -------------------------------------------------------------
# Simple 3D CNN for pitch classification - no ball tracking
# -------------------------------------------------------------

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

MEAN = torch.tensor([0.485, 0.456, 0.406])  # ImageNet statistics
STD  = torch.tensor([0.229, 0.224, 0.225])

# ---------- Simple Dataset (Video Only) ----------------------------------

class SimplePitchDataset(tud.Dataset):
    def __init__(self, num_frames: int = 16, resolution: int = 112):
        self.num_frames, self.res = num_frames, resolution
        
        # Initialize label mapping
        self.label2idx = {'fastball': 0, 'breaking': 1, 'offspeed': 2}
        
        # Load JSON data
        json_path = 'data/mlb-youtube-segmented.json' 
        with open(json_path, 'r', encoding='utf-8') as f:
            pitch_data = json.load(f)

        print(f"🚀 Creating simple video dataset (no ball tracking)...")

        # SIMPLIFIED PITCH GROUPING - only 3 categories
        def group_pitch_type(pitch_type):
            pitch_type = pitch_type.lower()
            if 'fastball' in pitch_type or 'sinker' in pitch_type or 'cutter' in pitch_type:
                return 'fastball'
            elif 'curve' in pitch_type or 'slider' in pitch_type:
                return 'breaking'
            else:  # changeup, knuckleball, etc.
                return 'offspeed'

        # Build samples from video directory
        self.samples = []
        video_dir = 'no_contact_pitches'
        
        if not os.path.exists(video_dir):
            print(f"❌ Video directory not found: {video_dir}")
            return
        
        print(f"📁 Scanning video directory: {video_dir}")
        
        # Get all video files
        video_files = [f for f in os.listdir(video_dir) if f.endswith('.mp4')]
        print(f"📹 Found {len(video_files)} video files")
        
        for video_file in video_files:
            video_id = video_file.split('.')[0]
            video_path = os.path.join(video_dir, video_file)
            
            if video_id in pitch_data:
                pitch_type = pitch_data[video_id]['type']
                grouped_type = group_pitch_type(pitch_type)
                self.samples.append([video_path, grouped_type, video_id])

        # Shuffle for good distribution
        random.shuffle(self.samples)
        
        print(f"✅ Created simple dataset with {len(self.samples)} samples")
        print(f"📊 Classes: {self.label2idx}")
        
        # Print class distribution
        class_counts = {'fastball': 0, 'breaking': 0, 'offspeed': 0}
        for _, label, _ in self.samples:
            class_counts[label] += 1
        
        print(f"📈 Class distribution:")
        for label, count in class_counts.items():
            percentage = count / len(self.samples) * 100
            print(f"   {label:>10}: {count:4d} samples ({percentage:5.1f}%)")

    def __len__(self): 
        return len(self.samples)

    def _sample_indices(self, num_total: int) -> list[int]:
        """Uniformly sample self.num_frames indices over [0, num_total)."""
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
            print(f"Error loading {path}: {e}")
            return self.__getitem__((idx + 1) % len(self.samples))

# ---------- Simple 3D CNN Model ------------------------------------------

class Simple3DCNNClassifier(nn.Module):
    def __init__(self, num_classes: int, dropout_rate: float = 0.5):
        super().__init__()
        
        # Use pretrained 3D ResNet as backbone
        self.backbone = torchvision.models.video.r3d_18(pretrained=True)
        
        # Get the feature dimension from the original FC layer
        feature_dim = self.backbone.fc.in_features
        
        # Replace the final layer with our classifier
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

# ---------- Training Functions -------------------------------------------

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
        
        # Gradient clipping for training stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optim.step()

        running_loss += loss.item() * labels.size(0)
        _, pred = out.max(1)
        correct += pred.eq(labels).sum().item()
        total += labels.size(0)
        
        if batch_idx % 20 == 0:  # Print progress
            print(f"  Batch {batch_idx:3d}, Loss: {loss.item():.4f}, Acc: {correct/total*100:.1f}%")

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
    """Print comprehensive classification analysis"""
    idx2label = {v: k for k, v in label2idx.items()}
    class_names = [idx2label[i] for i in range(len(label2idx))]
    
    print("\n" + "="*60)
    print("🎯 CLASSIFICATION ANALYSIS")
    print("="*60)
    
    true_labels = np.array(true_labels)
    pred_labels = np.array(pred_labels)
    
    accuracy = np.mean(true_labels == pred_labels)
    f1_macro = f1_score(true_labels, pred_labels, average='macro')
    f1_weighted = f1_score(true_labels, pred_labels, average='weighted')
    precision_macro = precision_score(true_labels, pred_labels, average='macro')
    recall_macro = recall_score(true_labels, pred_labels, average='macro')
    
    print(f"\n📊 PERFORMANCE METRICS:")
    print(f"   🎯 Accuracy:           {accuracy*100:6.2f}%")
    print(f"   📈 F1-Score (Macro):   {f1_macro*100:6.2f}%")
    print(f"   📈 F1-Score (Weighted):{f1_weighted*100:6.2f}%")
    print(f"   🎖️  Precision (Macro): {precision_macro*100:6.2f}%")
    print(f"   🔍 Recall (Macro):     {recall_macro*100:6.2f}%")
    
    print(f"\n📋 DETAILED CLASSIFICATION REPORT:")
    report = classification_report(true_labels, pred_labels, target_names=class_names, digits=3)
    print(report)
    
    cm = confusion_matrix(true_labels, pred_labels)
    print(f"\n🔀 CONFUSION MATRIX:")
    print(f"{'True \\ Pred':>12}", end="")
    for pred_name in class_names:
        print(f"{pred_name:>10}", end="")
    print()
    
    for i, true_name in enumerate(class_names):
        print(f"{true_name:>12}", end="")
        for j in range(len(class_names)):
            print(f"{cm[i,j]:>10}", end="")
        print()

# ---------- Main Training Function -----------------------------------

def main(args):
    warnings.filterwarnings("ignore", category=UserWarning)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Using device: {device}")

    # Simple dataset - video only
    full_ds = SimplePitchDataset(
        num_frames=args.frames, 
        resolution=args.res
    )
    
    if len(full_ds) < 10:
        print("❌ ERROR: Not enough valid samples! Check your data directories.")
        return
        
    train_ds, val_ds = split_dataset(full_ds, val_ratio=0.2, seed=2025)
    
    train_loader = tud.DataLoader(train_ds, batch_size=args.bsz, shuffle=True,
                                  num_workers=4, pin_memory=True)
    val_loader   = tud.DataLoader(val_ds, batch_size=args.bsz, shuffle=False,
                                  num_workers=4, pin_memory=True)

    num_classes = len(full_ds.label2idx)
    print(f"\n📊 Dataset: {len(full_ds)} clips | {num_classes} classes")
    print(f"   Train: {len(train_ds)}, Val: {len(val_ds)}")
    print(f"   Label mapping: {full_ds.label2idx}")

    # Simple 3D CNN Model
    model = Simple3DCNNClassifier(
        num_classes=num_classes, 
        dropout_rate=args.dropout
    )
    model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n🔧 Model: Simple3DCNNClassifier")
    print(f"   📏 Trainable parameters: {total_params:,}")
    print(f"   🎬 Video-only classification (no ball tracking)")

    # Class weighting for imbalanced data
    class_counts = defaultdict(int)
    for _, label_str, _ in full_ds.samples:
        class_counts[label_str] += 1
    
    print(f"\n📊 Class Distribution & Weighting:")
    total_samples = len(full_ds.samples)
    weights = []
    for i in range(num_classes):
        class_name = [k for k, v in full_ds.label2idx.items() if v == i][0]
        count = class_counts[class_name]
        percentage = count / total_samples * 100
        
        # Calculate inverse frequency weights
        weight = total_samples / (num_classes * count) if count > 0 else 1.0
        weights.append(weight)
        
        print(f"   {class_name:>10}: {count:4d} samples ({percentage:5.1f}%) - weight: {weight:.3f}")
    
    class_weights = torch.tensor(weights, dtype=torch.float).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Optimizer with scheduler
    optim = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optim, mode='max', factor=0.5, patience=3, verbose=True
    )

    # Training loop
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    best_acc = 0.0
    patience_counter = 0
    max_patience = 5
    
    print(f"\n🚀 Starting simple 3D CNN training!")
    print(f"   ⏱️  Expected training time: ~{args.epochs * 3} minutes")
    print(f"   📁 Checkpoints will be saved to: {args.out}")
    
    for epoch in range(1, args.epochs + 1):
        print(f"\n" + "="*50)
        print(f"📅 EPOCH {epoch}/{args.epochs}")
        print("="*50)
        
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optim, device)
        va_loss, va_acc = evaluate(model, val_loader, criterion, device)
        dt = time.time() - t0
        
        train_losses.append(tr_loss)
        val_losses.append(va_loss)
        train_accs.append(tr_acc)
        val_accs.append(va_acc)
        
        # Update learning rate scheduler
        scheduler.step(va_acc)
        current_lr = optim.param_groups[0]['lr']
        
        print(f"\n📈 EPOCH {epoch:02d} SUMMARY:")
        print(f"   🏋️  Train: loss={tr_loss:.4f}, acc={tr_acc*100:6.2f}%")
        print(f"   ✅ Val:   loss={va_loss:.4f}, acc={va_acc*100:6.2f}%")
        print(f"   ⏱️  Time: {dt:.1f}s | LR: {current_lr:.2e}")
        
        # Early stopping logic
        if va_acc > best_acc:
            best_acc = va_acc
            patience_counter = 0
            print(f"   🎯 ✨ New best validation accuracy: {best_acc*100:.2f}%")
            
            # Save best model
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
            print(f"   💾 Best model saved!")
        else:
            patience_counter += 1
            print(f"   ⏰ No improvement for {patience_counter}/{max_patience} epochs")
        
        # Detailed analysis every 5 epochs
        if epoch % 5 == 0:
            print(f"\n🔍 DETAILED VALIDATION ANALYSIS - EPOCH {epoch}")
            print("-"*40)
            evaluate(model, val_loader, criterion, device, full_ds.label2idx, detailed=True)
        
        # Early stopping check
        if patience_counter >= max_patience:
            print(f"\n🛑 Early stopping triggered after {patience_counter} epochs without improvement")
            break

    # Final evaluation
    print(f"\n" + "="*60)
    print("🎉 TRAINING COMPLETED!")
    print("="*60)
    print(f"🏆 Best validation accuracy: {best_acc*100:.2f}%")
    print(f"💾 Best model saved to: {args.out/'best_simple_3d_cnn.pt'}")
    
    # Final detailed analysis
    print(f"\n🔍 FINAL MODEL EVALUATION:")
    print("-"*40)
    
    # Load best model for final evaluation
    best_ckpt = torch.load(args.out / "best_simple_3d_cnn.pt", weights_only=False)
    model.load_state_dict(best_ckpt["state_dict"])
    
    final_results = evaluate(model, val_loader, criterion, device, full_ds.label2idx, detailed=True)
    
    print(f"\n✨ TRAINING SUMMARY:")
    print(f"   🎯 Final Accuracy: {final_results[1]*100:.2f}%")
    print(f"   📊 Total Epochs: {epoch}")
    print(f"   🎬 Simple 3D CNN (video-only)")
    print(f"   📁 Model saved to: {args.out}")

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
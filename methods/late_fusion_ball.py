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
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score, roc_curve, auc
from sklearn.preprocessing import label_binarize
import pandas as pd

MEAN = torch.tensor([0.485, 0.456, 0.406])
STD  = torch.tensor([0.229, 0.224, 0.225])

class FastBallTrackingPitchDataset(tud.Dataset):
    def __init__(self, num_frames: int = 16, resolution: int = 112, 
                 use_ball_tracking: bool = True, ball_features_file: str = "ball_features.pkl"):
        self.samples = []
        self.label2idx = {}
        self.num_frames, self.res = num_frames, resolution
        self.use_ball_tracking = use_ball_tracking
        
        self.ball_features = {}
        if self.use_ball_tracking:
            if os.path.exists(ball_features_file):
                with open(ball_features_file, 'rb') as f:
                    self.ball_features = pickle.load(f)
            else:
                self.use_ball_tracking = False

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

        all_candidates = []
        folders = ['no_contact_pitches']
        
        for folder_path in folders:
            if not os.path.exists(folder_path):
                continue
                
            for filename in os.listdir(folder_path):
                try:
                    file_path = os.path.join(folder_path, filename)
                    pitch_id = str(filename).split('.')[0]
                    
                    if pitch_id not in pitch_data:
                        continue
                    
                    if self.use_ball_tracking and pitch_id not in self.ball_features:
                        continue
                        
                    original_type = pitch_data[pitch_id]['type']
                    grouped_type = group_pitch_type(original_type)
                    
                    try:
                        frames, _, _ = read_video(file_path, pts_unit="sec")
                        if frames.shape[0] == 0:
                            continue
                    except:
                        continue
                    
                    all_candidates.append([file_path, grouped_type, pitch_id])
                    if grouped_type not in self.label2idx:
                        self.label2idx[grouped_type] = len(self.label2idx)
                        
                except Exception as e:
                    continue

        max_samples = len(all_candidates)
        random.shuffle(all_candidates)
        self.samples = all_candidates[:max_samples]

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
        path, label_str, pitch_id = self.samples[idx]
        
        try:
            frames, _, _ = read_video(path, pts_unit="sec")
            
            if frames.shape[0] == 0:
                return self.__getitem__((idx + 1) % len(self.samples))
                
            sel = self._sample_indices(frames.shape[0])
            clip = self._preprocess(frames[sel])
            
            if self.use_ball_tracking and pitch_id in self.ball_features:
                ball_features = torch.tensor(
                    self.ball_features[pitch_id]['features'], 
                    dtype=torch.float32
                )
            else:
                ball_features = torch.zeros(15, dtype=torch.float32)
            
            label_idx = torch.tensor(self.label2idx[label_str], dtype=torch.long)
            return clip, ball_features, label_idx
            
        except Exception as e:
            return self.__getitem__((idx + 1) % len(self.samples))

class EnhancedBallTrackingPitchClassifier(nn.Module):
    def __init__(self, num_classes: int, dropout_rate: float = 0.5, ball_feature_dim: int = 15):
        super().__init__()
        
        self.video_backbone = torchvision.models.video.r3d_18(pretrained=True)
        video_feat_dim = self.video_backbone.fc.in_features
        self.video_backbone.fc = nn.Identity()
        
        self.ball_branch = nn.Sequential(
            nn.Linear(ball_feature_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        self.fusion = nn.Sequential(
            nn.Linear(video_feat_dim + 32, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, video_clip, ball_features):
        video_feats = self.video_backbone(video_clip)
        ball_feats = self.ball_branch(ball_features)
        combined = torch.cat([video_feats, ball_feats], dim=1)
        output = self.fusion(combined)
        
        return output

def split_dataset(ds, val_ratio=0.2, seed=42):
    n = len(ds)
    idxs = list(range(n))
    random.Random(seed).shuffle(idxs)
    val_sz = int(n * val_ratio)
    return tud.Subset(ds, idxs[val_sz:]), tud.Subset(ds, idxs[:val_sz])

def train_one_epoch(model, loader, criterion, optim, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    
    for batch_idx, (clips, ball_features, labels) in enumerate(loader):
        clips = clips.to(device, non_blocking=True)
        ball_features = ball_features.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optim.zero_grad()
        out = model(clips, ball_features)
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
def evaluate(model, loader, criterion, device, label2idx=None, detailed=False, save_path=None):
    model.eval()
    loss, correct, total = 0.0, 0, 0
    
    all_preds, all_labels, all_probs = [], [], []
    
    for clips, ball_features, labels in loader:
        clips = clips.to(device)
        ball_features = ball_features.to(device)
        labels = labels.to(device)
        
        out = model(clips, ball_features)
        loss += criterion(out, labels).item() * labels.size(0)
        
        probs = F.softmax(out, dim=1)
        _, pred = out.max(1)
        correct += pred.eq(labels).sum().item()
        total += labels.size(0)
        
        if detailed:
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    if detailed and label2idx is not None:
        print_detailed_results(all_labels, all_preds, all_probs, label2idx, save_path)
        
    return loss / total, correct / total

def plot_training_curves(train_losses, val_losses, train_accs, val_accs, save_path=None):
    plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    epochs = range(1, len(train_losses) + 1)
    
    ax1.plot(epochs, train_losses, 'b-o', label='Training Loss', linewidth=2, markersize=4)
    ax1.plot(epochs, val_losses, 'r-s', label='Validation Loss', linewidth=2, markersize=4)
    ax1.set_title('Loss Curves', fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(epochs, [acc*100 for acc in train_accs], 'b-o', label='Training Acc', linewidth=2, markersize=4)
    ax2.plot(epochs, [acc*100 for acc in val_accs], 'r-s', label='Validation Acc', linewidth=2, markersize=4)
    ax2.set_title('Accuracy Curves', fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    ax3.plot(epochs, train_losses, 'b-', alpha=0.7, label='Train Loss')
    ax3.plot(epochs, val_losses, 'r-', alpha=0.7, label='Val Loss')
    ax3_twin = ax3.twinx()
    ax3_twin.plot(epochs, [acc*100 for acc in train_accs], 'g--', alpha=0.7, label='Train Acc')
    ax3_twin.plot(epochs, [acc*100 for acc in val_accs], 'orange', linestyle='--', alpha=0.7, label='Val Acc')
    ax3.set_title('Combined Learning Curves', fontweight='bold')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Loss', color='blue')
    ax3_twin.set_ylabel('Accuracy (%)', color='green')
    ax3.legend(loc='upper left')
    ax3_twin.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)
    
    train_val_diff = [abs(train_accs[i] - val_accs[i])*100 for i in range(len(epochs))]
    ax4.plot(epochs, train_val_diff, 'purple', linewidth=2, marker='o', markersize=4)
    ax4.axhline(y=5, color='red', linestyle='--', alpha=0.7, label='Overfitting Threshold (5%)')
    ax4.set_title('Overfitting Analysis', fontweight='bold')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Train-Val Accuracy Gap (%)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path / 'training_curves.png', dpi=300, bbox_inches='tight')
    
    plt.show()

def plot_confusion_matrix(cm, class_names, save_path=None):
    plt.figure(figsize=(10, 8))
    
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    annotations = []
    for i in range(cm.shape[0]):
        row = []
        for j in range(cm.shape[1]):
            count = cm[i, j]
            percent = cm_percent[i, j]
            text = f'{count}\n({percent:.1f}%)'
            row.append(text)
        annotations.append(row)
    
    sns.heatmap(cm, annot=annotations, fmt='', cmap='Blues', cbar=True,
                xticklabels=class_names, yticklabels=class_names,
                square=True, linewidths=0.5, cbar_kws={"shrink": .8})
    
    plt.title('Confusion Matrix\n(Count and Percentage)', fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('Predicted Label', fontweight='bold')
    plt.ylabel('True Label', fontweight='bold')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    
    if save_path:
        plt.savefig(save_path / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
    
    plt.show()

def plot_class_performance(true_labels, pred_labels, probs, class_names, save_path=None):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    precision_scores = precision_score(true_labels, pred_labels, average=None)
    recall_scores = recall_score(true_labels, pred_labels, average=None)
    f1_scores = f1_score(true_labels, pred_labels, average=None)
    
    unique, counts = np.unique(true_labels, return_counts=True)
    ax1.bar(range(len(class_names)), counts, color=['skyblue', 'lightcoral', 'lightgreen'])
    ax1.set_title('Class Distribution (True Labels)', fontweight='bold')
    ax1.set_xlabel('Pitch Type')
    ax1.set_ylabel('Count')
    ax1.set_xticks(range(len(class_names)))
    ax1.set_xticklabels(class_names, rotation=45)
    for i, count in enumerate(counts):
        ax1.text(i, count + max(counts)*0.01, str(count), ha='center', fontweight='bold')
    
    x = np.arange(len(class_names))
    width = 0.25
    
    ax2.bar(x - width, precision_scores, width, label='Precision', color='skyblue', alpha=0.8)
    ax2.bar(x, recall_scores, width, label='Recall', color='lightcoral', alpha=0.8)
    ax2.bar(x + width, f1_scores, width, label='F1-Score', color='lightgreen', alpha=0.8)
    
    ax2.set_title('Per-Class Metrics Comparison', fontweight='bold')
    ax2.set_xlabel('Pitch Type')
    ax2.set_ylabel('Score')
    ax2.set_xticks(x)
    ax2.set_xticklabels(class_names, rotation=45)
    ax2.legend()
    ax2.set_ylim(0, 1.1)
    
    for i, (p, r, f) in enumerate(zip(precision_scores, recall_scores, f1_scores)):
        ax2.text(i - width, p + 0.02, f'{p:.2f}', ha='center', fontweight='bold', fontsize=9)
        ax2.text(i, r + 0.02, f'{r:.2f}', ha='center', fontweight='bold', fontsize=9)
        ax2.text(i + width, f + 0.02, f'{f:.2f}', ha='center', fontweight='bold', fontsize=9)
    
    if len(class_names) <= 5:
        true_labels_bin = label_binarize(true_labels, classes=range(len(class_names)))
        
        colors = ['blue', 'red', 'green', 'orange', 'purple']
        for i in range(len(class_names)):
            if len(class_names) == 2 and i == 1:
                continue
                
            fpr, tpr, _ = roc_curve(true_labels_bin[:, i], probs[:, i])
            roc_auc = auc(fpr, tpr)
            
            ax3.plot(fpr, tpr, color=colors[i], linewidth=2,
                    label=f'{class_names[i]} (AUC = {roc_auc:.2f})')
        
        ax3.plot([0, 1], [0, 1], 'k--', alpha=0.5, linewidth=1)
        ax3.set_title('ROC Curves', fontweight='bold')
        ax3.set_xlabel('False Positive Rate')
        ax3.set_ylabel('True Positive Rate')
        ax3.legend(loc='lower right')
        ax3.grid(True, alpha=0.3)
    else:
        ax3.text(0.5, 0.5, 'ROC curves\nnot displayed\n(too many classes)', 
                ha='center', va='center', transform=ax3.transAxes, fontsize=12)
        ax3.set_title('ROC Curves', fontweight='bold')
    
    max_probs = np.max(probs, axis=1)
    correct_mask = (true_labels == pred_labels)
    
    ax4.hist(max_probs[correct_mask], bins=20, alpha=0.7, label='Correct', color='green', density=True)
    ax4.hist(max_probs[~correct_mask], bins=20, alpha=0.7, label='Incorrect', color='red', density=True)
    ax4.set_title('Prediction Confidence Distribution', fontweight='bold')
    ax4.set_xlabel('Max Probability')
    ax4.set_ylabel('Density')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path / 'class_performance.png', dpi=300, bbox_inches='tight')
    
    plt.show()

def plot_ball_feature_analysis(ball_features_dict, class_mapping, save_path=None):
    if not ball_features_dict:
        return
    
    all_features = []
    all_labels = []
    
    for video_id, data in ball_features_dict.items():
        features = data['features']
        pitch_type = data['pitch_type'].lower()
        
        if 'fastball' in pitch_type or 'sinker' in pitch_type or 'cutter' in pitch_type:
            label = 'fastball'
        elif 'curve' in pitch_type or 'slider' in pitch_type:
            label = 'breaking'
        else:
            label = 'offspeed'
        
        all_features.append(features)
        all_labels.append(label)
    
    features_array = np.array(all_features)
    labels_array = np.array(all_labels)
    
    feature_names = [
        'Mean X', 'Mean Y', 'Std X', 'Std Y', 'Min X', 'Max X', 'Min Y', 'Max Y',
        'Detection Rate', 'Avg Confidence', 'Mean Speed', 'Speed Std', 'Vertical Velocity',
        'Curvature', 'Vertical Displacement'
    ]
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    feature_stds = np.std(features_array, axis=0)
    sorted_indices = np.argsort(feature_stds)[::-1]
    
    ax1.bar(range(len(feature_names)), feature_stds[sorted_indices], color='skyblue', alpha=0.8)
    ax1.set_title('Feature Importance (by Standard Deviation)', fontweight='bold')
    ax1.set_xlabel('Features')
    ax1.set_ylabel('Standard Deviation')
    ax1.set_xticks(range(len(feature_names)))
    ax1.set_xticklabels([feature_names[i] for i in sorted_indices], rotation=45, ha='right')
    
    corr_matrix = np.corrcoef(features_array.T)
    im = ax2.imshow(corr_matrix, cmap='RdBu', vmin=-1, vmax=1, aspect='auto')
    ax2.set_title('Feature Correlation Matrix', fontweight='bold')
    ax2.set_xticks(range(len(feature_names)))
    ax2.set_yticks(range(len(feature_names)))
    ax2.set_xticklabels(feature_names, rotation=45, ha='right', fontsize=8)
    ax2.set_yticklabels(feature_names, fontsize=8)
    plt.colorbar(im, ax=ax2, shrink=0.8)
    
    unique_labels = np.unique(labels_array)
    colors = ['blue', 'red', 'green']
    
    class_separation = []
    for i, feature_name in enumerate(feature_names):
        class_means = [np.mean(features_array[labels_array == label, i]) for label in unique_labels]
        separation = np.std(class_means)
        class_separation.append(separation)
    
    top_features = np.argsort(class_separation)[-6:]
    
    for idx, feature_idx in enumerate(top_features):
        if idx >= 6:
            break
            
        subplot_idx = idx + 1
        if subplot_idx <= 2:
            continue
        
        if subplot_idx == 3:
            ax = ax3
        elif subplot_idx == 4:
            ax = ax4
        else:
            break
        
        for j, label in enumerate(unique_labels):
            mask = labels_array == label
            feature_values = features_array[mask, feature_idx]
            ax.hist(feature_values, bins=15, alpha=0.7, label=label, color=colors[j], density=True)
        
        ax.set_title(f'{feature_names[feature_idx]} Distribution', fontweight='bold')
        ax.set_xlabel(feature_names[feature_idx])
        ax.set_ylabel('Density')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path / 'ball_feature_analysis.png', dpi=300, bbox_inches='tight')
    
    plt.show()
    
    return {
        'feature_stds': feature_stds,
        'feature_names': feature_names,
        'class_separation': class_separation,
        'most_discriminative': [feature_names[i] for i in top_features]
    }

def print_detailed_results(true_labels, pred_labels, probs, label2idx, save_path=None):
    idx2label = {v: k for k, v in label2idx.items()}
    class_names = [idx2label[i] for i in range(len(label2idx))]
    
    true_labels = np.array(true_labels)
    pred_labels = np.array(pred_labels)
    probs = np.array(probs)
    
    accuracy = np.mean(true_labels == pred_labels)
    f1_macro = f1_score(true_labels, pred_labels, average='macro')
    f1_weighted = f1_score(true_labels, pred_labels, average='weighted')
    precision_macro = precision_score(true_labels, pred_labels, average='macro')
    recall_macro = recall_score(true_labels, pred_labels, average='macro')
    
    report = classification_report(true_labels, pred_labels, target_names=class_names, digits=3)
    
    cm = confusion_matrix(true_labels, pred_labels)
    
    if save_path:
        plot_confusion_matrix(cm, class_names, save_path)
        plot_class_performance(true_labels, pred_labels, probs, class_names, save_path)
    
    return {
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'confusion_matrix': cm,
        'class_names': class_names
    }

def main(args):
    warnings.filterwarnings("ignore", category=UserWarning)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    full_ds = FastBallTrackingPitchDataset(
        num_frames=args.frames, 
        resolution=args.res, 
        use_ball_tracking=args.ball_tracking,
        ball_features_file=args.ball_features_file
    )
    
    if len(full_ds) < 10:
        return
        
    train_ds, val_ds = split_dataset(full_ds, val_ratio=0.2, seed=2025)
    
    train_loader = tud.DataLoader(train_ds, batch_size=args.bsz, shuffle=True,
                                  num_workers=2, pin_memory=True)
    val_loader   = tud.DataLoader(val_ds, batch_size=args.bsz, shuffle=False,
                                  num_workers=2, pin_memory=True)

    num_classes = len(full_ds.label2idx)

    model = EnhancedBallTrackingPitchClassifier(
        num_classes=num_classes, 
        dropout_rate=args.dropout,
        ball_feature_dim=15
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
        optim, mode='max', factor=0.5, patience=2, verbose=True
    )

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    best_acc = 0.0
    patience_counter = 0
    max_patience = 3
    
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
        else:
            patience_counter += 1
        
        if epoch % 3 == 0:
            evaluate(model, val_loader, criterion, device, full_ds.label2idx, detailed=True, save_path=args.out)

        ckpt = {
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "optimizer": optim.state_dict(),
            "scheduler": scheduler.state_dict(),
            "label2idx": full_ds.label2idx,
            "args": vars(args),
            "train_losses": train_losses,
            "val_losses": val_losses,
            "train_accs": train_accs,
            "val_accs": val_accs,
            "class_weights": weights,
            "best_accuracy": best_acc
        }
        torch.save(ckpt, args.out / f"fast_checkpoint_epoch_{epoch:02d}.pt")

        if va_acc > best_acc - 1e-6:
            torch.save(ckpt, args.out / "best_fast_ball_tracking.pt")
        
        if patience_counter >= max_patience:
            break

    plot_training_curves(train_losses, val_losses, train_accs, val_accs, save_path=args.out)
    
    if full_ds.use_ball_tracking and full_ds.ball_features:
        feature_analysis = plot_ball_feature_analysis(full_ds.ball_features, full_ds.label2idx, save_path=args.out)
    
    best_ckpt = torch.load(args.out / "best_fast_ball_tracking.pt")
    model.load_state_dict(best_ckpt["state_dict"])
    
    final_results = evaluate(model, val_loader, criterion, device, full_ds.label2idx, detailed=True, save_path=args.out)
    
    summary = {
        'final_accuracy': final_results[1],
        'best_accuracy': best_acc,
        'total_epochs': epoch,
        'class_distribution': dict(class_counts),
        'model_parameters': total_params,
        'ball_tracking_enabled': args.ball_tracking,
        'training_curves': {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_accs': train_accs,
            'val_accs': val_accs
        }
    }
    
    with open(args.out / 'training_summary.json', 'w') as f:
        summary_json = {}
        for key, value in summary.items():
            if isinstance(value, (np.ndarray, np.integer, np.floating)):
                summary_json[key] = value.item() if hasattr(value, 'item') else value.tolist()
            elif isinstance(value, dict):
                summary_json[key] = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in value.items()}
            else:
                summary_json[key] = value
        json.dump(summary_json, f, indent=2)

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Fast Ball Tracking Pitch Classification")
    p.add_argument("--out",                type=Path, default=Path("./checkpoints"))
    p.add_argument("--epochs",             type=int, default=15)
    p.add_argument("--bsz",                type=int, default=8)
    p.add_argument("--lr",                 type=float, default=5e-4)
    p.add_argument("--frames",             type=int, default=16)
    p.add_argument("--res",                type=int, default=112)
    p.add_argument("--dropout",            type=float, default=0.5)
    p.add_argument("--ball-tracking",      action="store_true", help="Use precomputed ball features")
    p.add_argument("--ball-features-file", type=str, default="ball_features.pkl", help="Precomputed features file")
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    main(args)

#!/usr/bin/env python

import argparse, math, os, random, time, warnings
from pathlib import Path

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
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from collections import defaultdict
import matplotlib.patches as mpatches
import cv2

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

MEAN = torch.tensor([0.485, 0.456, 0.406])
STD  = torch.tensor([0.229, 0.224, 0.225])

def init_yolo_model():
    if not YOLO_AVAILABLE:
        return None
    
    try:
        model = YOLO("yolov8x.pt")
        model.classes = [32]
        model.conf = 0.25
        model.iou = 0.45
        return model
    except Exception as e:
        return None

def extract_ball_features(video_path, yolo_model):
    if yolo_model is None:
        return torch.zeros(12, dtype=torch.float32)
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return torch.zeros(12, dtype=torch.float32)
    
    ball_positions = []
    frame_idx = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            h, w = frame.shape[:2]
            cropped = frame[0:int(h*0.8), :]
            
            results = yolo_model.predict(cropped, conf=0.25, verbose=False)
            
            if len(results) > 0 and len(results[0].boxes) > 0:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                cls_np = results[0].boxes.cls.cpu().numpy()
                confs = results[0].boxes.conf.cpu().numpy()
                
                ball_detections = [(i, confs[i]) for i, cls_id in enumerate(cls_np) if int(cls_id) == 32]
                
                if ball_detections:
                    best_idx = max(ball_detections, key=lambda x: x[1])[0]
                    x1, y1, x2, y2 = boxes[best_idx]
                    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                    cx_norm = cx / w
                    cy_norm = cy / (h * 0.8)
                    ball_positions.append((frame_idx, cx_norm, cy_norm, confs[best_idx]))
            
            frame_idx += 1
    
    except Exception as e:
        pass
    finally:
        cap.release()
    
    return trajectory_to_features(ball_positions, frame_idx)

def trajectory_to_features(ball_positions, total_frames):
    if len(ball_positions) < 2:
        return torch.zeros(12, dtype=torch.float32)
    
    ball_positions.sort(key=lambda x: x[0])
    
    frames = np.array([pos[0] for pos in ball_positions])
    x_coords = np.array([pos[1] for pos in ball_positions])
    y_coords = np.array([pos[2] for pos in ball_positions])
    confidences = np.array([pos[3] for pos in ball_positions])
    
    features = []
    
    features.extend([
        np.mean(x_coords),
        np.mean(y_coords),
        np.std(x_coords),
        np.std(y_coords),
        len(ball_positions) / max(total_frames, 1),
        np.mean(confidences),
    ])
    
    if len(x_coords) >= 2:
        frame_diffs = np.diff(frames)
        frame_diffs[frame_diffs == 0] = 1
        
        x_velocity = np.diff(x_coords) / frame_diffs
        y_velocity = np.diff(y_coords) / frame_diffs
        
        features.extend([
            np.mean(x_velocity),
            np.mean(y_velocity),
            np.std(x_velocity),
            np.std(y_velocity),
        ])
    else:
        features.extend([0.0, 0.0, 0.0, 0.0])
    
    if len(x_coords) >= 3:
        dx = np.diff(x_coords)
        dy = np.diff(y_coords)
        if len(dx) >= 2:
            d2x = np.diff(dx)
            d2y = np.diff(dy)
            curvature = np.mean(np.sqrt(d2x**2 + d2y**2))
        else:
            curvature = 0.0
        features.append(curvature)
    else:
        features.append(0.0)
    
    if len(x_coords) > 0:
        features.append(y_coords[-1] - y_coords[0])
    else:
        features.append(0.0)
    
    features = features[:12]
    while len(features) < 12:
        features.append(0.0)
    
    return torch.tensor(features, dtype=torch.float32)

class BallTrackingPitchDataset(tud.Dataset):
    def __init__(self, num_frames: int = 16, resolution: int = 112, use_ball_tracking: bool = True):
        self.samples = []
        self.label2idx = {}
        self.num_frames, self.res = num_frames, resolution
        self.use_ball_tracking = use_ball_tracking
        
        if self.use_ball_tracking:
            self.yolo_model = init_yolo_model()
        else:
            self.yolo_model = None

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
                        
                    original_type = pitch_data[pitch_id]['type']
                    grouped_type = group_pitch_type(original_type)
                    
                    try:
                        frames, _, _ = read_video(file_path, pts_unit="sec")
                        if frames.shape[0] == 0:
                            continue
                    except:
                        continue
                    
                    all_candidates.append([file_path, grouped_type])
                    if grouped_type not in self.label2idx:
                        self.label2idx[grouped_type] = len(self.label2idx)
                        
                except Exception as e:
                    continue

        max_samples = 300
        if len(all_candidates) > max_samples:
            random.shuffle(all_candidates)
            self.samples = all_candidates[:max_samples]
        else:
            self.samples = all_candidates

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
        path, label_str = self.samples[idx]
        
        try:
            frames, _, _ = read_video(path, pts_unit="sec")
            
            if frames.shape[0] == 0:
                return self.__getitem__((idx + 1) % len(self.samples))
                
            sel = self._sample_indices(frames.shape[0])
            clip = self._preprocess(frames[sel])
            
            if self.use_ball_tracking and self.yolo_model:
                ball_features = extract_ball_features(path, self.yolo_model)
            else:
                ball_features = torch.zeros(12, dtype=torch.float32)
            
            label_idx = torch.tensor(self.label2idx[label_str], dtype=torch.long)
            return clip, ball_features, label_idx
            
        except Exception as e:
            return self.__getitem__((idx + 1) % len(self.samples))

class BallTrackingPitchClassifier(nn.Module):
    def __init__(self, num_classes: int, dropout_rate: float = 0.3, ball_feature_dim: int = 12):
        super().__init__()
        
        self.video_backbone = torchvision.models.video.r3d_18(pretrained=True)
        video_feat_dim = self.video_backbone.fc.in_features
        self.video_backbone.fc = nn.Identity()
        
        self.ball_branch = nn.Sequential(
            nn.Linear(ball_feature_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        self.fusion = nn.Sequential(
            nn.Linear(video_feat_dim + 32, 256),
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
    
    all_preds = []
    all_labels = []
    all_probs = []
    
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
        return print_detailed_results(all_labels, all_preds, all_probs, label2idx), loss / total, correct / total
        
    return loss / total, correct / total

def print_detailed_results(true_labels, pred_labels, probs, label2idx):
    idx2label = {v: k for k, v in label2idx.items()}
    class_names = [idx2label[i] for i in range(len(label2idx))]
    
    true_labels = np.array(true_labels)
    pred_labels = np.array(pred_labels)
    probs = np.array(probs)
    
    accuracy = np.mean(true_labels == pred_labels)
    
    f1_macro = f1_score(true_labels, pred_labels, average='macro')
    f1_micro = f1_score(true_labels, pred_labels, average='micro')
    f1_weighted = f1_score(true_labels, pred_labels, average='weighted')
    
    precision_macro = precision_score(true_labels, pred_labels, average='macro')
    precision_micro = precision_score(true_labels, pred_labels, average='micro')
    precision_weighted = precision_score(true_labels, pred_labels, average='weighted')
    
    recall_macro = recall_score(true_labels, pred_labels, average='macro')
    recall_micro = recall_score(true_labels, pred_labels, average='micro')
    recall_weighted = recall_score(true_labels, pred_labels, average='weighted')
    
    report = classification_report(true_labels, pred_labels, target_names=class_names, digits=3)
    
    for i, class_name in enumerate(class_names):
        mask = true_labels == i
        if np.any(mask):
            class_accuracy = np.mean(pred_labels[mask] == i)
            class_count = np.sum(mask)
            correct_count = np.sum(pred_labels[mask] == i)
            
            class_f1 = f1_score(true_labels == i, pred_labels == i)
            class_precision = precision_score(true_labels == i, pred_labels == i)
            class_recall = recall_score(true_labels == i, pred_labels == i)
    
    cm = confusion_matrix(true_labels, pred_labels)
    
    for i, true_name in enumerate(class_names):
        total_true = np.sum(cm[i, :])
        for j in range(len(class_names)):
            pass
        if total_true > 0:
            class_acc = cm[i,i] / total_true * 100
    
    max_probs = np.max(probs, axis=1)
    correct_mask = true_labels == pred_labels
    
    correct_confidences = max_probs[correct_mask]
    wrong_confidences = max_probs[~correct_mask]
    
    if len(pred_labels) > 0:
        top_indices = np.argsort(max_probs)[-min(5, len(max_probs)):]
        
        for idx in reversed(top_indices):
            true_name = idx2label[true_labels[idx]]
            pred_name = idx2label[pred_labels[idx]]
            conf = max_probs[idx] * 100
            status = "correct" if true_labels[idx] == pred_labels[idx] else "wrong"
    
    return {
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'confusion_matrix': cm,
        'class_names': class_names
    }

def plot_training_results(train_losses, val_losses, train_accs, val_accs, save_dir):
    plt.style.use('default')
    
    fig = plt.figure(figsize=(20, 12))
    
    colors = {'train': '#2E86AB', 'val': '#A23B72', 'accent': '#F18F01'}
    
    ax1 = plt.subplot(2, 3, 1)
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, 'o-', color=colors['train'], linewidth=2, markersize=6, label='Training Loss')
    plt.plot(epochs, val_losses, 's-', color=colors['val'], linewidth=2, markersize=6, label='Validation Loss')
    plt.title('Training & Validation Loss', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    ax2 = plt.subplot(2, 3, 2)
    plt.plot(epochs, [acc*100 for acc in train_accs], 'o-', color=colors['train'], linewidth=2, markersize=6, label='Training Accuracy')
    plt.plot(epochs, [acc*100 for acc in val_accs], 's-', color=colors['val'], linewidth=2, markersize=6, label='Validation Accuracy')
    plt.title('Training & Validation Accuracy', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    ax3 = plt.subplot(2, 3, 3)
    plt.scatter(train_losses, [acc*100 for acc in train_accs], color=colors['train'], s=100, alpha=0.7, label='Training')
    plt.scatter(val_losses, [acc*100 for acc in val_accs], color=colors['val'], s=100, alpha=0.7, label='Validation')
    plt.title('Loss vs Accuracy Relationship', fontsize=14, fontweight='bold')
    plt.xlabel('Loss')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    ax4 = plt.subplot(2, 3, 4)
    train_improvement = [(acc - train_accs[0])*100 for acc in train_accs]
    val_improvement = [(acc - val_accs[0])*100 for acc in val_accs]
    plt.plot(epochs, train_improvement, 'o-', color=colors['train'], linewidth=2, markersize=6, label='Training Improvement')
    plt.plot(epochs, val_improvement, 's-', color=colors['val'], linewidth=2, markersize=6, label='Validation Improvement')
    plt.title('Learning Progress (% Improvement)', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Improvement (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    ax5 = plt.subplot(2, 3, 5)
    gap = [(train_accs[i] - val_accs[i])*100 for i in range(len(epochs))]
    plt.plot(epochs, gap, 'o-', color=colors['accent'], linewidth=2, markersize=6, label='Train-Val Gap')
    plt.title('Overfitting Analysis', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy Gap (%)')
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    
    best_val_acc = max(val_accs) * 100
    best_epoch = val_accs.index(max(val_accs)) + 1
    final_val_acc = val_accs[-1] * 100
    final_train_acc = train_accs[-1] * 100
    avg_gap = np.mean([(train_accs[i] - val_accs[i])*100 for i in range(len(epochs))])
    
    summary_text = f"""
    TRAINING SUMMARY
    
    Best Validation Accuracy: {best_val_acc:.1f}%
    Best Epoch: {best_epoch}
    
    Final Train Accuracy: {final_train_acc:.1f}%
    Final Val Accuracy: {final_val_acc:.1f}%
    
    Average Train-Val Gap: {avg_gap:.1f}%
    
    Total Epochs: {len(epochs)}
    """
    
    ax6.text(0.1, 0.9, summary_text, transform=ax6.transAxes, fontsize=12,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle="round,pad=0.5", facecolor=colors['accent'], alpha=0.2))
    
    plt.tight_layout()
    plt.savefig(save_dir / 'training_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

def plot_confusion_matrix(cm, class_names, save_dir):
    plt.figure(figsize=(10, 8))
    
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                square=True, linewidths=0.5, cbar_kws={"shrink": .8})
    
    plt.title('Confusion Matrix (Normalized)', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
    plt.ylabel('True Label', fontsize=12, fontweight='bold')
    
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            plt.text(j+0.5, i+0.7, f'({cm[i,j]})', 
                    ha='center', va='center', fontsize=10, color='gray')
    
    plt.tight_layout()
    plt.savefig(save_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_performance_dashboard(metrics, save_dir):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    
    metrics_names = ['Accuracy', 'F1-Score', 'Precision', 'Recall']
    metrics_values = [metrics['accuracy'], metrics['f1_macro'], 
                     metrics['precision_macro'], metrics['recall_macro']]
    
    bars = ax1.bar(metrics_names, [m*100 for m in metrics_values], color=colors)
    ax1.set_title('Performance Metrics Overview', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Score (%)')
    ax1.set_ylim(0, 100)
    
    for bar, value in zip(bars, metrics_values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{value*100:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    class_names = metrics['class_names']
    cm = metrics['confusion_matrix']
    class_accuracies = [cm[i,i]/cm[i,:].sum()*100 if cm[i,:].sum() > 0 else 0 
                       for i in range(len(class_names))]
    
    bars2 = ax2.bar(class_names, class_accuracies, color=colors[:len(class_names)])
    ax2.set_title('Per-Class Accuracy', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_ylim(0, 100)
    
    for bar, acc in zip(bars2, class_accuracies):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{acc:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    sns.heatmap(metrics['confusion_matrix'], annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=ax3)
    ax3.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    
    class_totals = [metrics['confusion_matrix'][i,:].sum() for i in range(len(class_names))]
    pie = ax4.pie(class_totals, labels=class_names, autopct='%1.1f%%', 
                  colors=colors[:len(class_names)], startangle=90)
    ax4.set_title('Class Distribution', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_dir / 'performance_dashboard.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

def main(args):
    warnings.filterwarnings("ignore", category=UserWarning)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    full_ds = BallTrackingPitchDataset(
        num_frames=args.frames, 
        resolution=args.res, 
        use_ball_tracking=args.ball_tracking
    )
    
    if len(full_ds) < 10:
        return
        
    train_ds, val_ds = split_dataset(full_ds, val_ratio=0.2, seed=2025)
    
    train_loader = tud.DataLoader(train_ds, batch_size=args.bsz, shuffle=True,
                                  num_workers=0, pin_memory=True)
    val_loader   = tud.DataLoader(val_ds, batch_size=args.bsz, shuffle=False,
                                  num_workers=0, pin_memory=True)

    num_classes = len(full_ds.label2idx)

    model = BallTrackingPitchClassifier(
        num_classes=num_classes, 
        dropout_rate=args.dropout,
        ball_feature_dim=12
    )
    model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    criterion = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    best_acc = 0.0
    
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optim, device)
        va_loss, va_acc = evaluate(model, val_loader, criterion, device)
        dt = time.time() - t0
        
        train_losses.append(tr_loss)
        val_losses.append(va_loss)
        train_accs.append(tr_acc)
        val_accs.append(va_acc)
        
        evaluate(model, val_loader, criterion, device, full_ds.label2idx, detailed=True)

        ckpt = {
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "optimizer": optim.state_dict(),
            "label2idx": full_ds.label2idx,
            "args": vars(args),
            "train_losses": train_losses,
            "val_losses": val_losses,
            "train_accs": train_accs,
            "val_accs": val_accs
        }
        torch.save(ckpt, args.out / f"checkpoint_epoch_{epoch:02d}.pt")

        if va_acc > best_acc:
            best_acc = va_acc
            torch.save(ckpt, args.out / "best_ball_tracking.pt")

    plot_training_results(train_losses, val_losses, train_accs, val_accs, args.out)
    
    if best_acc > 0:
        checkpoint = torch.load(args.out / "best_ball_tracking.pt")
        model.load_state_dict(checkpoint['state_dict'])
        
        model.eval()
        all_preds, all_labels, all_probs = [], [], []
        
        with torch.no_grad():
            for clips, ball_features, labels in val_loader:
                clips = clips.to(device)
                ball_features = ball_features.to(device)
                out = model(clips, ball_features)
                probs = F.softmax(out, dim=1)
                _, pred = out.max(1)
                
                all_preds.extend(pred.cpu().numpy())
                all_labels.extend(labels.numpy())
                all_probs.extend(probs.cpu().numpy())
        
        final_metrics = print_detailed_results(all_labels, all_preds, all_probs, full_ds.label2idx)
        
        plot_confusion_matrix(final_metrics['confusion_matrix'], final_metrics['class_names'], args.out)
        create_performance_dashboard(final_metrics, args.out)

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Ball Tracking Enhanced Pitch Classification")
    p.add_argument("--out",            type=Path, default=Path("./checkpoints"))
    p.add_argument("--epochs",         type=int, default=5)
    p.add_argument("--bsz",            type=int, default=4)
    p.add_argument("--lr",             type=float, default=1e-3)
    p.add_argument("--frames",         type=int, default=16)
    p.add_argument("--res",            type=int, default=112)
    p.add_argument("--dropout",        type=float, default=0.3)
    p.add_argument("--ball-tracking",  action="store_true", help="Enable ball tracking features")
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    main(args)

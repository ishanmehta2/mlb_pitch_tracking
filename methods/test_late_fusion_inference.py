#!/usr/bin/env python
# test_late_fusion_inference.py

import argparse
import json
import random
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision.io import read_video
from torchvision import transforms as T
from typing import Tuple, Optional
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

CATEGORIES = ["fastball", "breaking", "offspeed"]
PITCH_MAP = {
    "fastball": "fastball",  "sinker": "fastball",  "cutter": "fastball",
    "curveball": "breaking", "slider": "breaking",
    "changeup": "offspeed",  "knucklecurve": "offspeed",
}

class LateFusionNet(nn.Module):
    def __init__(self, freeze_backbones=False):
        super().__init__()
        self.back3d = torchvision.models.video.r3d_18(weights="KINETICS400_V1")
        self.back3d.fc = nn.Identity(); dim3d = 512
        self.back2d = torchvision.models.resnet18(weights="IMAGENET1K_V1")
        self.back2d.fc = nn.Identity(); dim2d = 512
        if freeze_backbones:
            for p in self.back3d.parameters(): p.requires_grad=False
            for p in self.back2d.parameters(): p.requires_grad=False
        self.classifier = nn.Sequential(
            nn.Linear(dim3d+dim2d,256), nn.ReLU(inplace=True), nn.Dropout(0.5), nn.Linear(256,3)
        )
    def forward(self, vid, img):
        z3 = self.back3d(vid)
        z2 = self.back2d(img)
        return self.classifier(torch.cat([z3,z2],1))

class VideoProcessor:
    
    def __init__(self, 
                 num_frames: int = 16,
                 frame_stride: int = 2,
                 size: Tuple[int,int] = (128,128),
                 use_center_frame_for_2d: bool = True):
        self.num_frames = num_frames
        self.frame_stride = frame_stride
        self.size = size
        self.center_only = use_center_frame_for_2d
        
        self.resize = T.Resize(size)
        self.center_crop = T.CenterCrop(size) 
    
        self.mean3d = torch.tensor([0.485,0.456,0.406])[:,None,None,None]
        self.std3d  = torch.tensor([0.229,0.224,0.225])[:,None,None,None]
        self.mean2d = torch.tensor([0.485,0.456,0.406])[:,None,None]
        self.std2d  = torch.tensor([0.229,0.224,0.225])[:,None,None]
    
    def _temporal_sample(self, frames: torch.Tensor, random_sample: bool = False) -> torch.Tensor:
        total = frames.shape[0]
        need = self.num_frames * self.frame_stride
        
        if total == 0:
            raise ValueError("Zero-length video")
        
        if total < need:
            reps = np.ceil(need / total).astype(int)
            frames = torch.cat([frames] * reps, dim=0)
            total = frames.shape[0]
        
        if random_sample:
            start = random.randint(0, total - need)
        else:
            start = (total - need) // 2
        
        return frames[start:start+need:self.frame_stride]
    
    def process_video(self, video_path: Path) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        try:
            frames, _, _ = read_video(str(video_path), pts_unit="sec")
            if frames.shape[0] == 0:
                print(f"Warning: Empty video file: {video_path}")
                return None
                
            frames = frames.float() / 255.0
            clip = self._temporal_sample(frames, random_sample=False)  # T,H,W,C
            clip = clip.permute(3,0,1,2)  # C,T,H,W
         
            clip = self.center_crop(self.resize(clip))
            clip = (clip - self.mean3d) / self.std3d
            if self.center_only:
                img = clip[:, clip.shape[1]//2]
            else:
                img = clip.mean(dim=1)
            
            img = (img - self.mean2d) / self.std2d
            clip = clip.unsqueeze(0)  # 1,C,T,H,W
            img = img.unsqueeze(0)    # 1,C,H,W
            
            return clip, img
            
        except Exception as e:
            print(f"Error processing video {video_path}: {e}")
            return None

def load_model(model_path: str, device: str = "cuda") -> LateFusionNet:
    model = LateFusionNet(freeze_backbones=False)

    checkpoint = torch.load(model_path, map_location=device)
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    
    print(f"Model loaded from {model_path}")
    if 'val_acc' in checkpoint:
        print(f"Validation accuracy: {checkpoint['val_acc']:.2f}%")
    
    return model

def predict_single_video(model: LateFusionNet, 
                        video_path: Path, 
                        processor: VideoProcessor,
                        metadata: dict = None,
                        device: str = "cuda") -> dict:

    result = processor.process_video(video_path)
    if result is None:
        return None
    
    clip, img = result
    clip = clip.to(device)
    img = img.to(device)

    with torch.no_grad():
        logits = model(clip, img)
        probs = torch.softmax(logits, dim=1)
        pred_idx = torch.argmax(probs, dim=1).item()
        confidence = probs[0, pred_idx].item()

    true_label = None
    if metadata and video_path.stem in metadata:
        pitch_type = metadata[video_path.stem].get('type')
        true_label = PITCH_MAP.get(pitch_type)
    
    return {
        'video_path': str(video_path),
        'video_id': video_path.stem,
        'prediction': CATEGORIES[pred_idx],
        'confidence': confidence,
        'probabilities': {cat: probs[0, i].item() for i, cat in enumerate(CATEGORIES)},
        'true_label': true_label,
        'correct': true_label == CATEGORIES[pred_idx] if true_label else None
    }

def test_random_samples(model: LateFusionNet,
                       video_dir: Path,
                       processor: VideoProcessor,
                       metadata: dict = None,
                       num_samples: int = 10,
                       device: str = "cuda") -> list:
    """Test on random samples from a directory."""
    video_files = list(video_dir.glob("*.mp4"))
    
    if not video_files:
        print(f"No .mp4 files found in {video_dir}")
        return []
    num_samples = min(num_samples, len(video_files))
    sampled_videos = random.sample(video_files, num_samples)
    
    results = []
    print(f"\nTesting {num_samples} random samples from {video_dir}")
    print("="*80)
    
    for i, video_path in enumerate(sampled_videos, 1):
        print(f"\n[{i}/{num_samples}] Processing: {video_path.name}")
        
        result = predict_single_video(model, video_path, processor, metadata, device)
        
        if result:
            results.append(result)

            print(f"  Prediction: {result['prediction']:<10} (confidence: {result['confidence']:.3f})")
            print(f"  Probabilities:")
            for cat, prob in result['probabilities'].items():
                print(f"    - {cat:<10}: {prob:.3f}")
            
            if result['true_label']:
                status = "✓" if result['correct'] else "✗"
                print(f"  True label: {result['true_label']:<10} [{status}]")
    
    return results

def calculate_metrics(y_true, y_pred, labels):
    report = classification_report(y_true, y_pred, labels=labels, 
                                  target_names=CATEGORIES, 
                                  output_dict=True, zero_division=0)

    from collections import Counter
    true_counts = Counter(y_true)

    metrics = {
        'per_class': {},
        'macro_avg': report['macro avg'],
        'weighted_avg': report['weighted avg']
    }

    for cat in CATEGORIES:
        if cat in report:
            metrics['per_class'][cat] = {
                'precision': report[cat]['precision'],
                'recall': report[cat]['recall'],
                'f1-score': report[cat]['f1-score'],
                'support': true_counts.get(cat, 0)  
            }
    
    return metrics, report

def analyze_results(results: list):
    if not results:
        print("\nNo results to analyze.")
        return
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    pred_counts = {cat: 0 for cat in CATEGORIES}
    conf_by_cat = {cat: [] for cat in CATEGORIES}
    
    for result in results:
        pred = result['prediction']
        pred_counts[pred] += 1
        conf_by_cat[pred].append(result['confidence'])
 
    print("\nPrediction Distribution:")
    total = len(results)
    for cat in CATEGORIES:
        count = pred_counts[cat]
        pct = 100 * count / total
        avg_conf = np.mean(conf_by_cat[cat]) if conf_by_cat[cat] else 0
        print(f"  {cat:<10}: {count:3d} ({pct:5.1f}%) - avg confidence: {avg_conf:.3f}")

    results_with_labels = [r for r in results if r['true_label'] is not None]
    if results_with_labels:
        y_true = [r['true_label'] for r in results_with_labels]
        y_pred = [r['prediction'] for r in results_with_labels]
        metrics, report = calculate_metrics(y_true, y_pred, CATEGORIES)

        correct = sum(1 for r in results_with_labels if r['correct'])
        accuracy = 100 * correct / len(results_with_labels)
        print(f"\nOverall Accuracy: {accuracy:.1f}% ({correct}/{len(results_with_labels)})")

        print("\n" + "="*80)
        print("PER-CLASS METRICS")
        print("="*80)
        print(f"{'Class':<12} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>10}")
        print("-"*55)
        
        for cat in CATEGORIES:
            if cat in metrics['per_class']:
                m = metrics['per_class'][cat]
                print(f"{cat:<12} {m['precision']:10.4f} {m['recall']:10.4f} "
                      f"{m['f1-score']:10.4f} {m['support']:10d}")
        
        print("-"*55)

        print(f"{'Macro avg':<12} {metrics['macro_avg']['precision']:10.4f} "
              f"{metrics['macro_avg']['recall']:10.4f} "
              f"{metrics['macro_avg']['f1-score']:10.4f} "
              f"{len(results_with_labels):10d}")
        
        print(f"{'Weighted avg':<12} {metrics['weighted_avg']['precision']:10.4f} "
              f"{metrics['weighted_avg']['recall']:10.4f} "
              f"{metrics['weighted_avg']['f1-score']:10.4f} "
              f"{len(results_with_labels):10d}")

        print("\n" + "="*80)
        print("WEIGHTED AVERAGE METRICS SUMMARY")
        print("="*80)
        print(f"Weighted Precision: {metrics['weighted_avg']['precision']:.4f}")
        print(f"Weighted Recall:    {metrics['weighted_avg']['recall']:.4f}")
        print(f"Weighted F1-Score:  {metrics['weighted_avg']['f1-score']:.4f}")

        print("\n" + "="*80)
        print("CONFUSION MATRIX")
        print("="*80)
        cm = confusion_matrix(y_true, y_pred, labels=CATEGORIES)
        
        print("               Predicted")
        print("          ", end="")
        for cat in CATEGORIES:
            print(f"{cat[:8]:>10}", end="")
        print()
        
        for i, true_cat in enumerate(CATEGORIES):
            print(f"True {true_cat[:8]:>8}", end="")
            for j in range(len(CATEGORIES)):
                print(f"{cm[i, j]:>10}", end="")
            print()
        
        print("\n" + "="*80)
        print("PER-CLASS DETAILED BREAKDOWN")
        print("="*80)
        for cat in CATEGORIES:
            cat_results = [r for r in results_with_labels if r['true_label'] == cat]
            if cat_results:
                cat_correct = sum(1 for r in cat_results if r['correct'])
                cat_acc = 100 * cat_correct / len(cat_results)
                print(f"  {cat.upper()}:")
                print(f"    Samples: {len(cat_results):3d} | Correct: {cat_correct:3d} | "
                      f"Accuracy: {cat_acc:5.1f}%")
                if cat in metrics['per_class']:
                    m = metrics['per_class'][cat]
                    print(f"    Precision: {m['precision']:5.1%} | Recall: {m['recall']:5.1%} | "
                          f"F1: {m['f1-score']:5.1%}")

def main():
    parser = argparse.ArgumentParser(description="Test Late-Fusion model on no-contact pitches")
    parser.add_argument("--model_path", default="best_model.pth", help="Path to saved model")
    parser.add_argument("--video_dir", required=True, help="Directory containing test videos")
    parser.add_argument("--metadata_json", help="Optional: JSON file with true labels")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of random samples to test")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", default="cuda", help="Device (cuda/cpu)")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = "cpu"

    metadata = None
    if args.metadata_json:
        with open(args.metadata_json, 'r') as f:
            metadata = json.load(f)
        print(f"Loaded metadata from {args.metadata_json}")

    processor = VideoProcessor()

    model = load_model(args.model_path, device)

    video_dir = Path(args.video_dir)
    results = test_random_samples(model, video_dir, processor, metadata, 
                                 args.num_samples, device)

    analyze_results(results)

if __name__ == "__main__":
    main()

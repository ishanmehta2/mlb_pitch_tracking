import argparse, csv, math, os, random, time, warnings
from pathlib import Path

import torch, torch.nn as nn
import torch.utils.data as tud
import torchvision
from torchvision.io import read_video
import os
import json

MEAN = torch.tensor([0.485, 0.456, 0.406])
STD  = torch.tensor([0.229, 0.224, 0.225])


class PitchVideoDataset(tud.Dataset):
    def __init__(self, num_frames: int = 16, resolution: int = 112):
        self.samples = []                 # [(path, label_idx), ...]
        self.label2idx = {}
        self.num_frames, self.res = num_frames, resolution

        json_path = 'data/mlb-youtube-segmented.json' 

        # Open and load the JSON file
        with open(json_path, 'r', encoding='utf-8') as f:
            pitch_data = json.load(f)

        folder_path = 'baseline_data'

        # Loop through all files in the 'baseline' folder
        for filename in os.listdir(folder_path):
            try:
                file_path = os.path.join(folder_path, filename)
                pitch_id = str(filename).split('.')[0]
                pitch_type = pitch_data[pitch_id]['type']
                self.samples.append([file_path, pitch_type])
                if pitch_type not in self.label2idx:            # unseen string?
                        self.label2idx[pitch_type] = len(self.label2idx)  # assign next id
            except Exception as err:
                print("skipped")
            
        folder_path = 'clips'

        # Loop through all files in the 'baseline' folder
        for filename in os.listdir(folder_path):
            try:
                if len(self.samples) >= 60:
                    break
                file_path = os.path.join(folder_path, filename)
                pitch_id = str(filename).split('.')[0]
                pitch_type = pitch_data[pitch_id]['type']
                self.samples.append([file_path, pitch_type])
                if pitch_type not in self.label2idx:            # unseen string?
                        self.label2idx[pitch_type] = len(self.label2idx)  # assign next id
            except Exception as err:
                print("skipping")
    


    def __len__(self): return len(self.samples)

    def _sample_indices(self, num_total: int) -> list[int]:
        if num_total <= self.num_frames:
            # Pad by repeating last frame
            return list(range(num_total)) + [num_total - 1] * (self.num_frames - num_total)
        step = num_total / self.num_frames
        return [math.floor(i * step) for i in range(self.num_frames)]

    @torch.no_grad()
    def _preprocess(self, vid: torch.Tensor) -> torch.Tensor:
        vid = vid.permute(0, 3, 1, 2).float() / 255.0          # T C H W
        vid = torch.nn.functional.interpolate(
            vid, size=self.res, mode="bilinear", align_corners=False
        )                                                      # T C H W
        vid = (vid - MEAN[:, None, None]) / STD[:, None, None]
        return vid.permute(1, 0, 2, 3)                         # C T H W

    def __getitem__(self, idx: int):
        path, label_str = self.samples[idx]
        frames, _, _ = read_video(path, pts_unit="sec")
        sel  = self._sample_indices(frames.shape[0])
        clip = self._preprocess(frames[sel])         # (C T H W)

        label_idx = torch.tensor(self.label2idx[label_str], dtype=torch.long)
        return clip, label_idx


def split_dataset(ds, val_ratio=0.1, seed=42):
    n = len(ds)
    idxs = list(range(n))
    random.Random(seed).shuffle(idxs)
    val_sz = int(n * val_ratio)
    return tud.Subset(ds, idxs[val_sz:]), tud.Subset(ds, idxs[:val_sz])


def train_one_epoch(model, loader, criterion, optim, scaler, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for clips, labels in loader:
        clips  = clips.to(device, non_blocking=True)
        labels = torch.as_tensor(labels, device=device)


        with torch.cuda.amp.autocast():
            out = model(clips)
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
    for clips, labels in loader:
        clips, labels = clips.to(device), labels.to(device)
        out = model(clips)
        loss += criterion(out, labels).item() * labels.size(0)
        _, pred = out.max(1)
        correct += pred.eq(labels).sum().item()
        total += labels.size(0)
        print(_, pred)
    return loss / total, correct / total


def main(args):
    warnings.filterwarnings("ignore", category=UserWarning)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    full_ds = PitchVideoDataset(num_frames=args.frames, resolution=args.res)
    train_ds, val_ds = split_dataset(full_ds, val_ratio=0.1, seed=2025)
    train_loader = tud.DataLoader(train_ds, batch_size=args.bsz, shuffle=True,
                                  num_workers=args.workers, pin_memory=True)
    val_loader   = tud.DataLoader(val_ds, batch_size=args.bsz, shuffle=False,
                                  num_workers=args.workers, pin_memory=True)

    num_classes = len(full_ds.label2idx)
    print(f"Dataset: {len(full_ds)} clips | {num_classes} classes")

    model = torchvision.models.video.r3d_18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optim     = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler    = torch.cuda.amp.GradScaler()

    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optim,
                                          scaler, device)
        va_loss, va_acc = evaluate(model, val_loader, criterion, device)
        dt = time.time() - t0
        print(f"[{epoch:02d}/{args.epochs}] "
              f"train loss {tr_loss:.3f} acc {tr_acc*100:5.1f}% | "
              f"val loss {va_loss:.3f} acc {va_acc*100:5.1f}% | "
              f"{dt/60:.1f} min")

        if va_acc > best_acc:
            best_acc = va_acc
            ckpt = {"epoch": epoch,
                    "state_dict": model.state_dict(),
                    "label2idx": full_ds.label2idx}
            torch.save(ckpt, args.out / "best.pt")

    print(f"Best val accuracy: {best_acc*100:.2f}%")
    print(f"Checkpoint saved to {args.out/'best.pt'}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out",   type=Path, default=Path("./checkpoints"))
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--bsz",    type=int, default=4)
    p.add_argument("--lr",     type=float, default=1e-4)
    p.add_argument("--frames", type=int, default=16)
    p.add_argument("--res",    type=int, default=112)
    p.add_argument("--workers", type=int, default=os.cpu_count()//2)
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    main(args)

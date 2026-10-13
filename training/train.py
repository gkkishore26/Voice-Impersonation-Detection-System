"""
Training script for Voice Impersonation Detection CNN

Usage:
    python train.py                     # default: 25 epochs, batch 32
    python train.py --epochs 15 --batch 64
"""
import os
import sys
import json
import time
import argparse
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import VoiceCNN, count_parameters
from preprocess import audio_to_input, CLASS_NAMES

ROOT = r"C:\Users\Kishore GK\Desktop\VIDS"
RAW = os.path.join(ROOT, 'dataset', 'mega')   # MEGA corpus (WaveFake8 + MLAAD + Libri + HF + field)
MODEL_DIR = os.path.join(ROOT, 'model')
EVAL_DIR = os.path.join(ROOT, 'evaluation')


# ---------------- Dataset ----------------
class VoiceDataset(Dataset):
    def __init__(self, augment=False):
        self.items = []
        self.augment = augment
        for cls_idx, cls in enumerate(CLASS_NAMES):        # fake=0, real=1
            d = os.path.join(RAW, cls)
            if not os.path.isdir(d):
                continue
            for fn in os.listdir(d):
                if fn.lower().endswith(('.wav', '.mp3', '.flac', '.m4a', '.ogg')):
                    self.items.append((os.path.join(d, fn), cls_idx))
        if not self.items:
            raise RuntimeError(f'No audio found in {RAW} - dataset not downloaded yet')

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, label = self.items[i]
        try:
            mel = audio_to_input(path)
        except Exception as e:
            print(f'  [warn] failed to load {path}: {e}')
            mel = np.zeros((1, 128, 128), dtype=np.float32)
        x = torch.from_numpy(mel)
        # ---- augmentation (training only): noise + gain + SpecAugment ----
        # SpecAugment (Park et al.): random freq/time masks force the CNN to
        # use distributed patterns instead of memorizing local artifacts.
        if self.augment:
            if torch.rand(1).item() < 0.3:                 # additive noise
                x = x + torch.randn_like(x) * 0.03
            if torch.rand(1).item() < 0.3:                 # random gain
                x = x * (0.8 + 0.4 * torch.rand(1).item())
            if torch.rand(1).item() < 0.5:                 # freq mask (<=16 bins)
                f = torch.randint(0, 16, (1,)).item()
                f0 = torch.randint(0, max(128 - f, 1), (1,)).item()
                x[:, f0:f0 + f, :] = 0.0
            if torch.rand(1).item() < 0.5:                 # time mask (<=24 frames)
                t = torch.randint(0, 24, (1,)).item()
                t0 = torch.randint(0, max(128 - t, 1), (1,)).item()
                x[:, :, t0:t0 + t] = 0.0
            x = torch.clamp(x, 0.0, 1.0)
        return x, label


# ---------------- Metrics ----------------
def evaluate(model, loader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    tot_loss = correct = total = 0
    tp = tn = fp = fn = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            tot_loss += criterion(logits, y).item() * len(y)
            pred = logits.argmax(1)
            correct += (pred == y).sum().item()
            total += len(y)
            # class 1 = REAL, class 0 = FAKE
            tp += ((pred == 1) & (y == 1)).sum().item()   # real correctly called real
            tn += ((pred == 0) & (y == 0)).sum().item()   # fake correctly called fake
            fp += ((pred == 1) & (y == 0)).sum().item()   # fake called real (bad)
            fn += ((pred == 0) & (y == 1)).sum().item()   # real called fake (bad)
    acc = 100.0 * correct / max(total, 1)
    return {
        'loss': tot_loss / max(total, 1), 'acc': acc,
        'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
        'precision': tp / max(tp + fp, 1),
        'recall': tp / max(tp + fn, 1),
        'f1': 2 * tp / max(2 * tp + fp + fn, 1),
        'n': total,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=25)
    ap.add_argument('--batch', type=int, default=32)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--val-split', type=float, default=0.2)
    args = ap.parse_args()

    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(EVAL_DIR, exist_ok=True)

    device = ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}' + (f' -> {torch.cuda.get_device_name(0)}' if device == 'cuda' else ' (CPU - slower!)'))
    print(f'Model: Professor K')

    # ---- data ----
    full = VoiceDataset(augment=True)
    n_val = max(1, int(len(full) * args.val_split))
    n_train = len(full) - n_val
    train_set, val_set = random_split(full, [n_train, n_val],
                                      generator=torch.Generator().manual_seed(42))
    # validation set must NOT be augmented
    val_set.dataset.augment = False

    print(f'Train: {n_train}  Val: {n_val}  (real+fake total: {len(full)})')
    train_loader = DataLoader(train_set, batch_size=args.batch, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=args.batch, shuffle=False, num_workers=0)

    # ---- model ----
    model = VoiceCNN().to(device)
    print(f'Parameters: {count_parameters(model):,}')
    # class-weighted loss: keeps majority class from dominating the boundary
    counts = np.bincount([full.items[i][1] for i in range(len(full))], minlength=2)
    cw = torch.tensor([len(full) / (2 * max(int(c), 1)) for c in counts],
                      dtype=torch.float32).to(device)
    print(f'class counts [fake, real] = {counts.tolist()} '
          f'loss weights = {[round(float(x), 3) for x in cw]}')
    criterion = nn.CrossEntropyLoss(weight=cw)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

    best_acc = 0.0
    history = []
    print('\n---- training start ----')
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        run_loss = correct = total = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            run_loss += loss.item() * len(y)
            correct += (logits.argmax(1) == y).sum().item()
            total += len(y)

        train_acc = 100.0 * correct / max(total, 1)
        train_loss = run_loss / max(total, 1)
        val = evaluate(model, val_loader, device)
        scheduler.step(val['loss'])
        dt = time.time() - t0
        history.append({'epoch': epoch, 'train_loss': train_loss, 'train_acc': train_acc,
                        'val_loss': val['loss'], 'val_acc': val['acc']})
        print(f'Epoch {epoch:2d}/{args.epochs} | '
              f'train loss {train_loss:.4f} acc {train_acc:5.1f}% | '
              f'val loss {val["loss"]:.4f} acc {val["acc"]:5.1f}% | '
              f'{dt:.1f}s')

        if val['acc'] >= best_acc:
            best_acc = val['acc']
            torch.save({'model_state': model.state_dict(), 'epoch': epoch,
                        'val_acc': best_acc, 'class_names': CLASS_NAMES,
                        'model_name': 'Professor K'},
                       os.path.join(MODEL_DIR, 'professor_k.pth'))

    # ---- final evaluation on validation set ----
    print(f'\nBest val accuracy: {best_acc:.1f}%')
    final = evaluate(model, val_loader, device)
    print(f'Final metrics: acc={final["acc"]:.1f}%  '
          f'precision={final["precision"]:.3f}  recall={final["recall"]:.3f}  f1={final["f1"]:.3f}')
    print(f'Confusion: TP(real ok)={final["tp"]}  TN(fake ok)={final["tn"]}  '
          f'FP(fake->real!)={final["fp"]}  FN(real->fake!)={final["fn"]}')

    with open(os.path.join(MODEL_DIR, 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    with open(os.path.join(MODEL_DIR, 'metrics.json'), 'w') as f:
        json.dump(final, f, indent=2)
    print('Saved: model/professor_k.pth, training_history.json, metrics.json')


if __name__ == '__main__':
    main()

"""FINAL training round: fine-tune MEGA model on group-split field data.

  - train/val split BY SOURCE FILE (no leakage): val = held-out fra voice
    + 5 held-out libri speakers (see prep_final.py)
  - starts from current professor_k.pth (MEGA), low LR, weighted loss
  - checkpoints best val; saves final + history; updates field manifest log

Usage: python train_final.py --epochs 12 --lr 1e-4
"""
import os, sys, json, argparse, shutil, time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

ROOT = r"C:\Users\Kishore GK\Desktop\VIDS"
sys.path.insert(0, os.path.join(ROOT, 'training'))
import train as T
from model import VoiceCNN
from preprocess import CLASS_NAMES

FINAL = os.path.join(ROOT, 'dataset', 'final')
MODEL_DIR = os.path.join(ROOT, 'model')
CKPT = os.path.join(MODEL_DIR, 'professor_k.pth')


def make_loader(sub, augment, batch):
    T.RAW = os.path.join(FINAL, sub)
    ds = T.VoiceDataset(augment=augment)
    return ds, DataLoader(ds, batch_size=batch,
                          shuffle=(sub == 'train'), num_workers=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=12)
    ap.add_argument('--batch', type=int, default=32)
    ap.add_argument('--lr', type=float, default=1e-4)
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}' + (f' -> {torch.cuda.get_device_name(0)}' if device == 'cuda' else ''))
    if device != 'cuda':
        print('REFUSING CPU training.')
        return

    train_ds, train_loader = make_loader('train', True, args.batch)
    val_ds, val_loader = make_loader('val', False, args.batch)
    print(f'FINAL round data: train={len(train_ds)} val={len(val_ds)} (file-group split)')

    ckpt = torch.load(CKPT, map_location=device, weights_only=False)
    base_epoch = ckpt.get('epoch', 0)
    model = VoiceCNN().to(device)
    model.load_state_dict(ckpt['model_state'])

    counts = np.bincount([train_ds.items[i][1] for i in range(len(train_ds))], minlength=2)
    w = torch.tensor([len(train_ds) / (2 * max(int(c), 1)) for c in counts],
                     dtype=torch.float32).to(device)
    print(f'class counts [fake, real] = {counts.tolist()}')
    criterion = nn.CrossEntropyLoss(weight=w)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best = 0.0
    print('\n---- FINAL training start ----')
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, rl, c, tot = time.time(), 0.0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            rl += loss.item() * len(y)
            c += (logits.argmax(1) == y).sum().item()
            tot += len(y)
        val = T.evaluate(model, val_loader, device)
        print(f'FINAL {epoch:2d}/{args.epochs} | train loss {rl/max(tot,1):.4f} '
              f'acc {100*c/max(tot,1):5.1f}% | val loss {val["loss"]:.4f} '
              f'acc {val["acc"]:5.1f}% | {time.time()-t0:.1f}s', flush=True)
        if val['acc'] >= best:
            best = val['acc']
            torch.save({'model_state': model.state_dict(), 'epoch': base_epoch + epoch,
                        'val_acc': best, 'class_names': CLASS_NAMES,
                        'model_name': 'Professor K',
                        'finetuned_from_epoch': base_epoch, 'final_round': True}, CKPT)

    final = T.evaluate(model, val_loader, device)
    print(f'\nBest val: {best:.1f}% | final: acc={final["acc"]:.1f}% '
          f'prec={final["precision"]:.3f} rec={final["recall"]:.3f} f1={final["f1"]:.3f}')
    json.dump({'epochs': args.epochs, 'lr': args.lr, 'base_epoch': base_epoch,
               'best_val_acc': best, 'final': final},
              open(os.path.join(MODEL_DIR, 'final_history.json'), 'w'), indent=1)
    print('Saved: model/professor_k.pth (FINAL), final_history.json')


if __name__ == '__main__':
    main()

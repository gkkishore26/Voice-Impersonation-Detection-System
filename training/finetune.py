"""Fine-tune Professor K on field data (user-labeled uploads + rehearsal).

Genuine supervised fine-tuning (not a description file):
  - loads model/professor_k.pth (backs it up once -> professor_k_v1_backup.pth)
  - trains a few epochs, low LR, class-weighted loss, augmentation on
  - validates on held-out split, saves new checkpoint + history
  - appends results to dataset/field/manifest.json (progressive learning log)

Usage: python finetune.py --epochs 10 --lr 1e-4
"""
import os, sys, json, argparse, shutil, time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

ROOT = r"C:\Users\Kishore GK\Desktop\VIDS"
sys.path.insert(0, os.path.join(ROOT, 'training'))
import train as T
from model import VoiceCNN
from preprocess import CLASS_NAMES

FT = os.path.join(ROOT, 'dataset', 'finetune')
MODEL_DIR = os.path.join(ROOT, 'model')
CKPT = os.path.join(MODEL_DIR, 'professor_k.pth')
BACKUP = os.path.join(MODEL_DIR, 'professor_k_v1_backup.pth')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=10)
    ap.add_argument('--batch', type=int, default=32)
    ap.add_argument('--lr', type=float, default=1e-4)
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}' + (f' -> {torch.cuda.get_device_name(0)}' if device == 'cuda' else ''))
    if device != 'cuda':
        print('REFUSING CPU training (user demanded GPU).')
        return

    if not os.path.exists(CKPT):
        print('No base model found - run training/train.py first.')
        return
    if not os.path.exists(BACKUP):
        shutil.copy(CKPT, BACKUP)
        print('backed up v1 -> professor_k_v1_backup.pth')

    # point VoiceDataset at the finetune mix
    T.RAW = FT
    full = T.VoiceDataset(augment=True)
    n_val = max(1, int(len(full) * 0.2))
    train_set, val_set = random_split(full, [len(full) - n_val, n_val],
                                      generator=torch.Generator().manual_seed(7))
    val_set.dataset.augment = False
    print(f'Finetune data: train={len(train_set)} val={len(val_set)} (total {len(full)})')

    ckpt = torch.load(CKPT, map_location=device, weights_only=False)
    base_epoch = ckpt.get('epoch', 0)
    model = VoiceCNN().to(device)
    model.load_state_dict(ckpt['model_state'])

    # class weights from actual counts (fake-heavy mix safe)
    counts = np.bincount([full.items[i][1] for i in range(len(full))], minlength=2)
    w = torch.tensor([len(full) / (2 * max(c, 1)) for c in counts],
                     dtype=torch.float32).to(device)
    print(f'class counts [fake, real] = {counts.tolist()} weights = {[round(float(x),3) for x in w]}')
    criterion = nn.CrossEntropyLoss(weight=w)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    train_loader = DataLoader(train_set, batch_size=args.batch, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=args.batch, shuffle=False, num_workers=0)

    best = 0.0
    print('\n---- fine-tuning start ----')
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
        print(f'FT Epoch {epoch:2d}/{args.epochs} | train loss {rl/max(tot,1):.4f} '
              f'acc {100*c/max(tot,1):5.1f}% | val loss {val["loss"]:.4f} '
              f'acc {val["acc"]:5.1f}% | {time.time()-t0:.1f}s')
        if val['acc'] >= best:
            best = val['acc']
            torch.save({'model_state': model.state_dict(), 'epoch': base_epoch + epoch,
                        'val_acc': best, 'class_names': CLASS_NAMES,
                        'model_name': 'Professor K',
                        'finetuned_from_epoch': base_epoch,
                        'field': True}, CKPT)

    final = T.evaluate(model, val_loader, device)
    print(f'\nBest ft val acc: {best:.1f}% | final: acc={final["acc"]:.1f}% '
          f'prec={final["precision"]:.3f} rec={final["recall"]:.3f} f1={final["f1"]:.3f}')
    json.dump({'epochs': args.epochs, 'lr': args.lr, 'base_epoch': base_epoch,
               'best_val_acc': best, 'final': final,
               'field_sources': sorted({m.get('source', '?') for m in
                                        json.load(open(os.path.join(ROOT, "dataset", "field", "manifest.json")))}),
               }, open(os.path.join(MODEL_DIR, 'finetune_history.json'), 'w'), indent=1)
    print('Saved: model/professor_k.pth (fine-tuned), finetune_history.json')


if __name__ == '__main__':
    main()

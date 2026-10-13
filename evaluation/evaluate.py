"""
Evaluation: test the trained CNN and generate report-ready outputs

Outputs (evaluation/):
    confusion_matrix.png      - report figure
    training_curves.png       - loss + accuracy curves
    metrics.json              - accuracy/precision/recall/F1/ROC-AUC + probe

Honesty notes (read before quoting numbers):
  * val split shares base clips/speakers with train (stitched segments +
    single-speaker LJSpeech lineage) -> val metrics are OPTIMISTIC.
  * same_corpus_probe reuses the HF corpus that also fed training
    (different segments, shared base clips) -> also optimistic.
  * The true generalization check is field testing (fresh uploads:
    ElevenLabs, mic recordings) in the web app.
"""
import os
import sys
import json
import glob
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

ROOT = r"C:\Users\Kishore GK\Desktop\VIDS"
MODEL_PATH = os.path.join(ROOT, 'model', 'professor_k.pth')
EVAL_DIR = os.path.join(ROOT, 'evaluation')

# import from training/ folder
sys.path.insert(0, os.path.join(ROOT, 'training'))
from model import VoiceCNN
from train import VoiceDataset, evaluate
from preprocess import CLASS_NAMES, audio_to_input

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot_confusion(cm, acc, path):
    fig, ax = plt.subplots(figsize=(5.6, 4.8))
    im = ax.imshow(cm, cmap='Blues')
    labels = ['AI-Generated\n(fake)', 'Real Human\n(real)']
    ax.set_xticks([0, 1], labels, fontsize=10)
    ax.set_yticks([0, 1], labels, fontsize=10)
    ax.set_xlabel('Predicted Label', fontsize=11)
    ax.set_ylabel('True Label', fontsize=11)
    ax.set_title(f'Confusion Matrix  (Accuracy {acc:.1f}%)', fontsize=12, fontweight='bold')
    thresh = cm.max() / 2 if cm.max() else 0
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f'{cm[i, j]}', ha='center', va='center',
                    fontsize=22, fontweight='bold',
                    color='white' if cm[i, j] > thresh else '#1a1a1a')
    fig.colorbar(im, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print('saved', path)


def plot_curves(history, path):
    ep = [h['epoch'] for h in history]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

    ax1.plot(ep, [h['train_acc'] for h in history], 'b-o', ms=4, label='Train')
    ax1.plot(ep, [h['val_acc'] for h in history], 'g-o', ms=4, label='Validation')
    ax1.set_title('Model Accuracy', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Epoch'); ax1.set_ylabel('Accuracy (%)')
    ax1.legend(); ax1.grid(alpha=.4)

    ax2.plot(ep, [h['train_loss'] for h in history], 'b-o', ms=4, label='Train')
    ax2.plot(ep, [h['val_loss'] for h in history], 'g-o', ms=4, label='Validation')
    ax2.set_title('Model Loss', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Epoch'); ax2.set_ylabel('Loss')
    ax2.legend(); ax2.grid(alpha=.4)

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print('saved', path)


def probe_same_corpus(model, device):
    """Run the model over dataset/balanced (HF corpus).

    Shares base clips with training -> OPTIMISTIC by construction.
    Returns dict with accuracy/f1/roc_auc/n. Never raises.
    """
    from sklearn.metrics import roc_auc_score, f1_score
    base = os.path.join(ROOT, 'dataset', 'balanced')
    files, labels = [], []
    for cls, y in (('fake', 1), ('real', 0)):
        for p in sorted(glob.glob(os.path.join(base, cls, '*.wav'))):
            files.append(p)
            labels.append(y)
    y_true, y_score, n_ok = [], [], 0
    model.eval()
    with torch.no_grad():
        for p, y in zip(files, labels):
            try:
                mel = audio_to_input(p)
            except Exception:
                continue
            x = torch.from_numpy(mel).unsqueeze(0).to(device)
            pr = float(torch.softmax(model(x), dim=1)[0][0])
            y_true.append(y)
            y_score.append(pr)
            n_ok += int((pr >= 0.5) == bool(y))
    n = len(y_true)
    if n == 0:
        return {'accuracy': None, 'f1': None, 'roc_auc': None, 'n': 0}
    pred = [int(s >= 0.5) for s in y_score]
    try:
        auc = round(float(roc_auc_score(y_true, y_score)), 4)
    except Exception:
        auc = None
    return {'accuracy': round(100.0 * n_ok / n, 2),
            'f1': round(float(f1_score(y_true, pred, zero_division=0)), 4),
            'roc_auc': auc, 'n': n}


def main():
    os.makedirs(EVAL_DIR, exist_ok=True)

    if not os.path.exists(MODEL_PATH):
        print('No trained model found. Run training/train.py first.')
        return

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ckpt = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    model = VoiceCNN()
    model.load_state_dict(ckpt['model_state'])
    model.to(device).eval()
    print(f'Model: epoch {ckpt.get("epoch")}  val_acc {ckpt.get("val_acc")}%  device {device}')

    # same split as training (seed 42) for honest evaluation
    full = VoiceDataset(augment=False)
    n_val = max(1, int(len(full) * 0.2))
    n_train = len(full) - n_val
    _, val_set = random_split(full, [n_train, n_val],
                              generator=torch.Generator().manual_seed(42))
    loader = DataLoader(val_set, batch_size=64, shuffle=False)
    print(f'Evaluating on {len(val_set)} samples')

    m = evaluate(model, loader, device)
    cm = np.array([[m['tn'], m['fp']],
                   [m['fn'], m['tp']]])

    # ---- ROC-AUC needs per-sample scores: collect them ----
    from sklearn.metrics import roc_auc_score
    y_true, y_score = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            probs = torch.softmax(model(x), dim=1)[:, 0].cpu().numpy()  # P(fake)
            y_score.extend(probs.tolist())
            y_true.extend((y.numpy() == 0).astype(int).tolist())  # 1 = fake
    try:
        auc = float(roc_auc_score(y_true, y_score))
    except Exception:
        auc = None

    print(f"\nAccuracy : {m['acc']:.2f}%")
    print(f"Precision: {m['precision']:.3f}")
    print(f"Recall   : {m['recall']:.3f}")
    print(f"F1-score : {m['f1']:.3f}")
    print(f"ROC-AUC  : {auc if auc is None else round(auc, 4)}")
    print(f"Confusion matrix:\n{cm}")

    # ---- same-corpus probe: HF set (shares base clips with train -> optimistic,
    #       labelled as such; true test = fresh field uploads) ----
    probe = probe_same_corpus(model, device)
    print(f"\nSame-corpus probe (HF set, shares base clips - optimistic): "
          f"acc={probe['accuracy']}% f1={probe['f1']} auc={probe['roc_auc']} n={probe['n']}")

    results = {
        'accuracy': round(m['acc'], 2),
        'precision': round(m['precision'], 4),
        'recall': round(m['recall'], 4),
        'f1': round(m['f1'], 4),
        'roc_auc': round(auc, 4) if auc is not None else None,
        'confusion_matrix': cm.tolist(),
        'n_samples': m['n'],
        'same_corpus_probe': probe,
        'leakage_note': ('val + probe share base clips/speakers with train '
                         '(stitched overlap, single-speaker LJSpeech lineage); '
                         'treat as optimistic. Field uploads = true test.'),
        'epoch': ckpt.get('epoch'),
        'model_val_acc': ckpt.get('val_acc'),
        'class_names': CLASS_NAMES,
    }
    with open(os.path.join(EVAL_DIR, 'metrics.json'), 'w') as f:
        json.dump(results, f, indent=2)

    plot_confusion(cm, m['acc'], os.path.join(EVAL_DIR, 'confusion_matrix.png'))

    hist_path = os.path.join(ROOT, 'model', 'training_history.json')
    if os.path.exists(hist_path):
        with open(hist_path) as f:
            history = json.load(f)
        plot_curves(history, os.path.join(EVAL_DIR, 'training_curves.png'))

    print('\nDone. Outputs in evaluation/')


if __name__ == '__main__':
    main()

"""FILE-LEVEL honest test on strictly held-out files (never trained).

TEST SET (mirrors deployment: whole-file verdict via chunk pipeline):
  fake: male ai.mp3, femail ai.mp3, ai_gen_por_*, ai_gen_ita_* (2 held-out voices)
  real: 10 LibriSpeech speakers not present in field/ (fresh dirs)
Metrics: file-level acc/precision/recall/F1 + per-file table.
"""
import os, glob, json
import numpy as np
import torch

import sys
sys.path.insert(0, r"C:\Users\Kishore GK\Desktop\VIDS\training")
from model import VoiceCNN
from preprocess import audio_to_input, audio_to_chunks, ffmpeg_load, SR

ROOT = r"C:\Users\Kishore GK\Desktop\VIDS"
DL = r"C:\Users\Kishore GK\Downloads"
LIBRI = os.path.join(ROOT, 'dataset', 'kaggle', 'librispeech', 'LibriSpeech')
FIELD = os.path.join(ROOT, 'dataset', 'field')
CKPT = os.path.join(ROOT, 'model', 'professor_k.pth')
T = 2.5  # same calibration as deployment


def file_verdict(model, device, path):
    """Whole-file verdict exactly like the app: chunk + majority vote."""
    import librosa
    try:
        dur = librosa.get_duration(path=path)
    except Exception:
        dur = len(ffmpeg_load(path, SR)) / SR
    if dur <= 15:
        mel = audio_to_input(path)
        x = torch.from_numpy(mel).unsqueeze(0).to(device)
        with torch.no_grad():
            pr = torch.softmax(model(x) / T, dim=1)[0].cpu().numpy()
        pf, prl = float(pr[0]), float(pr[1])
        return (prl >= pf), max(prl, pf), 1
    chunks = audio_to_chunks(path)
    votes, confs = [], []
    xs = torch.from_numpy(np.stack([m for _, m in chunks])).to(device)
    with torch.no_grad():
        allp = torch.softmax(model(xs) / T, dim=1).cpu().numpy()
    for pr in allp:
        pf, prl = float(pr[0]), float(pr[1])
        votes.append(prl >= pf)
        confs.append(max(prl, pf))
    real_votes = sum(votes)
    overall = real_votes >= len(votes) / 2
    return overall, (real_votes / len(votes) if overall
                     else 1 - real_votes / len(votes)), len(votes)


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ckpt = torch.load(CKPT, map_location=device, weights_only=False)
    model = VoiceCNN().to(device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    print(f"model epoch {ckpt.get('epoch')} on {device}")

    # held-out fake files (fra voice never trained; male/female never trained)
    fake_files = [os.path.join(DL, 'male ai.mp3'), os.path.join(DL, 'femail ai.mp3')]
    fake_files += sorted(glob.glob(os.path.join(DL, 'ai_gen_fra_*.wav')))
    # held-out real: libri speakers absent from field manifest
    manifest = json.load(open(os.path.join(FIELD, 'manifest.json')))
    used = {m['source'].split('-')[0] for m in manifest if m['source'][0].isdigit()}
    used |= {'1089', '1188', '121', '1221'}
    real_files, seen = [], set()
    for p in sorted(glob.glob(os.path.join(LIBRI, '*', '*', '*', '*.flac'))):
        spk = p.split(os.sep)[-3]
        if spk not in used and spk not in seen:
            seen.add(spk)
            real_files.append(p)
        if len(real_files) >= 10:
            break

    print(f'TEST files: {len(fake_files)} fake + {len(real_files)} real (all held out)')
    rows = []
    for p, y in [(f, 0) for f in fake_files] + [(f, 1) for f in real_files]:
        try:
            is_real, conf, n = file_verdict(model, device, p)
        except Exception as e:
            print('  SKIP', os.path.basename(p)[:40], e)
            continue
        ok = (is_real == bool(y))
        rows.append((os.path.basename(p)[:44], 'real' if y else 'fake',
                     'real' if is_real else 'fake', round(conf * 100, 1), n, ok))
        print(f"  [{'OK ' if ok else 'MISS'}] {rows[-1][0]:46s} true={rows[-1][1]:4s} "
              f"pred={rows[-1][2]:4s} ({rows[-1][3]}%, {n} segs)")

    yt = [1 if r[1] == 'real' else 0 for r in rows]
    yp = [1 if r[2] == 'real' else 0 for r in rows]
    tp = sum(1 for t, p in zip(yt, yp) if t == 1 and p == 1)
    tn = sum(1 for t, p in zip(yt, yp) if t == 0 and p == 0)
    fp = sum(1 for t, p in zip(yt, yp) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(yt, yp) if t == 1 and p == 0)
    acc = (tp + tn) / max(len(rows), 1)
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    out = {'n_files': len(rows), 'accuracy': round(acc, 4),
           'precision': round(prec, 4), 'recall': round(rec, 4),
           'f1': round(f1, 4), 'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
           'files': [{'file': r[0], 'true': r[1], 'pred': r[2],
                      'conf': r[3], 'segments': r[4], 'correct': r[5]} for r in rows]}
    json.dump(out, open(os.path.join(ROOT, 'evaluation', 'file_test.json'), 'w'), indent=1)
    print(f"\nFILE-LEVEL TEST: acc={acc:.1%} prec={prec:.3f} rec={rec:.3f} f1={f1:.3f} "
          f"(TP={tp} TN={tn} FP={fp} FN={fn}, n={len(rows)})")


if __name__ == '__main__':
    main()

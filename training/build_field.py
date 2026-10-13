"""Build progressive field dataset from user-provided labeled uploads.

Input : user files with known labels (here: 2 ElevenLabs AI fakes)
Output: dataset/field/fake/*.wav (3s slices) + manifest.json
        dataset/finetune/{fake,real}/ (field + rehearsal mix, balanced)

Rehearsal (old multi clips mixed in) prevents catastrophic forgetting:
the model learns the NEW voices WITHOUT unlearning the 8 vocoders.
Future labeled uploads append here via --extra "path label".
"""
import os, glob, json, random, shutil
import numpy as np
import soundfile as sf

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from preprocess import SR

ROOT = r"C:\Users\Kishore GK\Desktop\VIDS"
FIELD = os.path.join(ROOT, 'dataset', 'field')
FT = os.path.join(ROOT, 'dataset', 'finetune')
MULTI = os.path.join(ROOT, 'dataset', 'multi')
TARGET_N = SR * 3

# user-provided labeled uploads: (path, label)
UPLOADS = [
    (r"C:\Users\Kishore GK\Downloads\ElevenLabs_2026-10-11T03_38_34_Russ – Deep, Smooth and Articulate_pvc_sp110_s50_sb75_se0_b_m2.mp3", 'fake'),
    (r"C:\Users\Kishore GK\Downloads\motivation is a myth.mp3", 'fake'),
    (r"C:\Users\Kishore GK\Downloads\real_test_1.spk1089.mp3", 'real'),
    (r"C:\Users\Kishore GK\Downloads\real_test_2.spk1188.mp3", 'real'),
    (r"C:\Users\Kishore GK\Downloads\real_test_3.spk121.mp3", 'real'),
    (r"C:\Users\Kishore GK\Downloads\real_test_4.spk1221.mp3", 'real'),
]

LIBRI_REAL_EXTRA = 25   # extra LibriSpeech speakers as ground-truth real

REHEARSE_FAKE = 100   # multi fake clips mixed in (anti-forgetting)
REHEARSE_REAL = 200   # multi real clips (negatives)


def load_full(p):
    from preprocess import ffmpeg_load
    try:
        import librosa
        y, _ = librosa.load(p, sr=SR, mono=True)
        return y
    except Exception:
        return ffmpeg_load(p, SR)


def slice_3s(y):
    return [y[i:i + TARGET_N] for i in range(0, len(y) - TARGET_N + 1, TARGET_N)]


def main():
    os.makedirs(os.path.join(FIELD, 'fake'), exist_ok=True)
    os.makedirs(os.path.join(FIELD, 'real'), exist_ok=True)
    manifest_path = os.path.join(FIELD, 'manifest.json')
    manifest = json.load(open(manifest_path)) if os.path.exists(manifest_path) else []

    # extra LibriSpeech speakers (ground-truth bonafide, skip test speakers)
    libri_all = sorted(glob.glob(os.path.join(
        ROOT, 'dataset', 'kaggle', 'librispeech', 'LibriSpeech', '*', '*', '*', '*.flac')))
    seen_spk = {'1089', '1188', '121', '1221'}
    extra, used = [], set()
    for p in libri_all:
        spk = p.split(os.sep)[-3]
        if spk not in seen_spk and spk not in used:
            used.add(spk)
            extra.append((p, 'real'))
        if len(extra) >= LIBRI_REAL_EXTRA:
            break

    # ---- 1. slice user uploads into labeled field segments ----
    n_new = 0
    for src, label in UPLOADS + extra:
        if not os.path.exists(src):
            print(f'missing file (skipped): {os.path.basename(src)[:50]}')
            continue
        if any(m.get('source') == os.path.basename(src) for m in manifest):
            print(f'skip (already ingested): {os.path.basename(src)[:50]}')
            continue
        y = load_full(src)
        tag = ''.join(c if c.isalnum() else '_' for c in os.path.basename(src)[:24]).strip('_')
        for i, seg in enumerate(slice_3s(y)):
            m = np.abs(seg).max()
            if m > 1e-6:
                seg = seg / m * 0.95
            name = f'field_{tag}_{i:03d}.wav'
            sf.write(os.path.join(FIELD, label, name), seg, SR)
            manifest.append({'file': name, 'label': label,
                             'source': os.path.basename(src), 'start_s': i * 3})
            n_new += 1
        print(f'{os.path.basename(src)[:50]:52s} -> {len(slice_3s(y))} segments [{label}]')
    json.dump(manifest, open(manifest_path, 'w'), indent=1)
    nf = len([m for m in manifest if m['label'] == 'fake'])
    nr = len([m for m in manifest if m['label'] == 'real'])
    print(f'field total: fake={nf} real={nr} (+{n_new} new)')

    # ---- 2. assemble balanced finetune mix (field + rehearsal) ----
    for cls in ('fake', 'real'):
        d = os.path.join(FT, cls)
        if os.path.exists(d):
            shutil.rmtree(d)
        os.makedirs(d)
    field_fake = sorted(glob.glob(os.path.join(FIELD, 'fake', '*.wav')))
    field_real = sorted(glob.glob(os.path.join(FIELD, 'real', '*.wav')))
    multi_fake = sorted(glob.glob(os.path.join(MULTI, 'fake', '*.wav')))
    multi_real = sorted(glob.glob(os.path.join(MULTI, 'real', '*.wav')))
    random.seed(11)
    rehearse_f = random.sample(multi_fake, min(REHEARSE_FAKE, len(multi_fake)))
    rehearse_r = random.sample(multi_real, min(REHEARSE_REAL, len(multi_real)))
    for p in field_fake + rehearse_f:
        shutil.copy(p, os.path.join(FT, 'fake', os.path.basename(p)))
    for p in field_real + rehearse_r:
        shutil.copy(p, os.path.join(FT, 'real', os.path.basename(p)))
    nf = len(os.listdir(os.path.join(FT, 'fake')))
    nr = len(os.listdir(os.path.join(FT, 'real')))
    print(f'finetune mix: fake={nf} ({len(field_fake)} field + {len(rehearse_f)} rehearsal) '
          f'real={nr} ({len(field_real)} field + {len(rehearse_r)} rehearsal) | -> dataset/finetune/')


if __name__ == '__main__':
    main()

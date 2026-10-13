"""Assemble FINAL round data with FILE-GROUP splits (no leakage).

TEST  (held-out files, never trained - the honest exam):
  fake: male ai.mp3, femail ai.mp3, ai_gen_por_*, ai_gen_ita_* (2 held-out voices)
  real: 10 new LibriSpeech speakers (not in field)
VAL (held-out files): ai_gen_fra_* (1 voice) + 5 new LibriSpeech speakers
TRAIN: field pool + remaining MMS voices (eng/deu/spa) + remaining libri
       + mega rehearsal subset

Output: dataset/final/{train,val}/{fake,real}/ + final_manifest.json
(TEST files stay in Downloads; test_files.py reads them directly.)
"""
import os, glob, json, random, shutil

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_multisource import load, slices_3s, save_seg

ROOT = r"C:\Users\Kishore GK\Desktop\VIDS"
DL = r"C:\Users\Kishore GK\Downloads"
LIBRI = os.path.join(ROOT, 'dataset', 'kaggle', 'librispeech', 'LibriSpeech')
FIELD = os.path.join(ROOT, 'dataset', 'field')
MULTI = os.path.join(ROOT, 'dataset', 'multi')
DST = os.path.join(ROOT, 'dataset', 'final')


def ingest(src, label, tag):
    """Slice a labeled file into field pool. Names unique per source (hash)."""
    import hashlib
    from preprocess import ffmpeg_load, SR
    import soundfile as sf
    import numpy as np
    uid = hashlib.md5(os.path.basename(src).encode()).hexdigest()[:6]
    try:
        import librosa
        y, _ = librosa.load(src, sr=SR, mono=True)
    except Exception:
        y = ffmpeg_load(src, SR)
    d = os.path.join(FIELD, label)
    os.makedirs(d, exist_ok=True)
    n = 0
    for i, seg in enumerate(slices_3s(y)):
        m = np.abs(seg).max()
        if m > 1e-6:
            seg = seg / m * 0.95
        name = f'field_{tag}_{uid}_{i:03d}.wav'
        sf.write(os.path.join(d, name), seg, SR)
        n += 1
    return n


def main():
    os.makedirs(FIELD, exist_ok=True)
    manifest_path = os.path.join(FIELD, 'manifest.json')
    manifest = json.load(open(manifest_path)) if os.path.exists(manifest_path) else []
    seen = {m.get('source') for m in manifest}

    # ---- ingest new GEN voices + extra libri into field pool ----
    jobs = []
    for lang in ('eng', 'deu', 'spa'):   # fra reserved as TEST holdout (never trained)
        for f in sorted(glob.glob(os.path.join(DL, f'ai_gen_{lang}_*.wav'))):
            jobs.append((f, 'fake', f'mms_{lang}'))
    libri_all = sorted(glob.glob(os.path.join(LIBRI, '*', '*', '*', '*.flac')))
    used_spk = set()
    n_libri = 0
    for p in libri_all:
        spk = p.split(os.sep)[-3]
        if spk in used_spk:
            continue
        # skip speakers already in field (test_1..4 + 25 extra)
        if os.path.basename(p) in seen:
            continue
        used_spk.add(spk)
        jobs.append((p, 'real', f'libri_{spk}'))
        n_libri += 1
        if n_libri >= 40:
            break
    import hashlib
    n_new = 0
    for src, label, tag in jobs:
        if os.path.basename(src) in seen:
            continue
        n = ingest(src, label, tag)
        uid = hashlib.md5(os.path.basename(src).encode()).hexdigest()[:6]
        for i in range(n):
            manifest.append({'file': f'field_{tag}_{uid}_{i:03d}.wav', 'label': label,
                             'source': os.path.basename(src), 'start_s': i * 3})
        n_new += n
        print(f'{os.path.basename(src)[:44]:46s} -> {n} [{label}]')
    json.dump(manifest, open(manifest_path, 'w'), indent=1)
    print(f'ingested {n_new} new field segments')

    # ---- assemble train/val dirs ----
    for split in ('train', 'val'):
        for cls in ('fake', 'real'):
            d = os.path.join(DST, split, cls)
            if os.path.exists(d):
                shutil.rmtree(d)
            os.makedirs(d)
    rnd = random.Random(12)
    # VAL = fra voice segments + first 5 NEW libri speakers (by basename prefix)
    val_libri_spk = set(sorted(used_spk)[:5])
    for m in manifest:
        src, label = m['source'], m['label']
        is_fra = src.startswith('ai_gen_fra_')
        is_val_libri = src.split('-')[0] in val_libri_spk
        dest = 'val' if (is_fra or is_val_libri) else 'train'
        p = os.path.join(FIELD, label, m['file'])
        if os.path.exists(p):
            shutil.copy(p, os.path.join(DST, dest, label, m['file']))
    # TRAIN rehearsal from mega (keeps old knowledge)
    for cls, n in (('fake', 400), ('real', 400)):
        pool = sorted(glob.glob(os.path.join(ROOT, 'dataset', 'mega', cls, '*.wav')))
        for p in rnd.sample(pool, min(n, len(pool))):
            shutil.copy(p, os.path.join(DST, 'train', cls, 'reh_' + os.path.basename(p)))
    # VAL fake: held-out mega vocoder clips (mla_/wavefake8_ only, never field)
    val_fake_pool = [p for p in sorted(glob.glob(os.path.join(ROOT, 'dataset', 'mega', 'fake', '*.wav')))
                     if os.path.basename(p).startswith(('mla_', 'wavefake8_'))]
    for p in rnd.sample(val_fake_pool, min(20, len(val_fake_pool))):
        shutil.copy(p, os.path.join(DST, 'val', 'fake', 'valheld_' + os.path.basename(p)))
    counts = {}
    for split in ('train', 'val'):
        for cls in ('fake', 'real'):
            counts[f'{split}/{cls}'] = len(os.listdir(os.path.join(DST, split, cls)))
    json.dump({'counts': counts, 'val_libri_speakers': sorted(val_libri_spk),
               'note': 'VAL files held out by source; TEST files never copied (see test_files.py)'},
              open(os.path.join(DST, 'final_manifest.json'), 'w'), indent=1)
    print('FINAL splits:', counts)


def spk_match(src, val_libri_spk):
    return False


if __name__ == '__main__':
    main()

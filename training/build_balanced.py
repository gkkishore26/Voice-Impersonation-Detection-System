"""
Build a DURATION-BALANCED training dataset.

Problem in raw dataset: fake clips are ~1.6s (mostly silence-padded to 3s),
real clips are ~7s full speech. The CNN learns 'silence = fake' shortcut.

Fix: create uniform 3-second segments where BOTH classes have full speech:
  - real : cut into consecutive 3s slices (full speech)
  - fake : concatenate neighbouring short clips back-to-back until ~3s,
           so fake segments also contain ~3s of actual speech.

Output: dataset/balanced/real/*.wav, dataset/balanced/fake/*.wav  (all ~3s)
"""
import os
import glob
import numpy as np
import soundfile as sf
from tqdm import tqdm

ROOT = r"C:\Users\Kishore GK\Desktop\VIDS"
SRC = os.path.join(ROOT, 'dataset', 'raw')
DST = os.path.join(ROOT, 'dataset', 'balanced')
SR = 22050
TARGET = 3.0                      # seconds per segment
TARGET_N = int(SR * TARGET)
MIN_SEG = int(SR * 2.0)           # keep segment only if >= 2s speech


def load(p):
    y, sr = sf.read(p, dtype='float32')
    if y.ndim > 1:
        y = y.mean(axis=1)
    if sr != SR:
        n = int(len(y) * SR / sr)
        y = np.interp(np.linspace(0, len(y) - 1, n), np.arange(len(y)), y).astype('float32')
    return y


def slices(y, size):
    """consecutive non-overlapping slices of `size` samples (drop tail)."""
    for i in range(0, len(y) - size + 1, size):
        yield y[i:i + size]


def build_class(cls):
    outdir = os.path.join(DST, cls)
    os.makedirs(outdir, exist_ok=True)
    files = sorted([p for p in glob.glob(os.path.join(SRC, cls, '*'))
                    if not p.endswith('.DS_Store')])

    # ---- collect all speech audio for this class ----
    waves = []
    for p in files:
        try:
            y = load(p)
            if len(y) >= TARGET_N:           # already a full 3s clip
                waves.append(y)
            elif len(y) >= SR * 0.5:         # short clip (>=0.5s) still usable for stitching
                waves.append(y)
        except Exception as e:
            print('  skip', os.path.basename(p), e)

    out = []
    idx = 0

    if cls == 'real':
        # ---- real: slice every long clip into 3s parts ----
        for y in waves:
            for s in slices(y, TARGET_N):
                if len(s) == TARGET_N:
                    out.append(s)
    else:
        # ---- fake: stitch short clips back-to-back until 3s ----
        buf = []
        buf_len = 0
        for y in waves:
            buf.append(y); buf_len += len(y)
            while buf_len >= TARGET_N:
                cat = np.concatenate(buf)
                out.append(cat[:TARGET_N])
                # keep remainder as start of next buffer
                rest = cat[TARGET_N:]
                buf, buf_len = [rest] if len(rest) else [], len(rest)
        # leftover tail (>=2s) also usable
        if buf_len >= MIN_SEG:
            cat = np.concatenate(buf)
            pad = np.zeros(TARGET_N - len(cat), dtype='float32')
            out.append(np.concatenate([cat, pad]))

    # ---- save ----
    for seg in out:
        # peak-normalise like real recording would be
        m = np.abs(seg).max()
        if m > 1e-6:
            seg = seg / m * 0.95
        sf.write(os.path.join(outdir, f'{cls}_{idx:04d}.wav'), seg, SR)
        idx += 1
    return idx


def main():
    print('Building duration-balanced dataset (all segments = 3.0s)...\n')
    for cls in ('real', 'fake'):
        n = build_class(cls)
        print(f'{cls:5s}: {n} segments saved -> dataset/balanced/{cls}/')
    print('\nDone. Train on this for honest accuracy.')


if __name__ == '__main__':
    main()

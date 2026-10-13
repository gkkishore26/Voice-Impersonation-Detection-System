"""Build MULTI-SOURCE balanced dataset for generalization.

Fake: 8 WaveFake vocoders (~200 full-speech 3s segments each) + HF fakes
Real: LJSpeech slices + HF balanced real slices
Every segment = 3.0s @22050Hz full speech. Caps keep classes AND
sources balanced so no single generator dominates.

Output: dataset/multi/{real,fake}/*.wav + sources.json manifest
"""
import os, glob, json
import numpy as np
import soundfile as sf

ROOT = r"C:\Users\Kishore GK\Desktop\VIDS"
WF = os.path.join(ROOT, 'dataset', 'kaggle', 'wavefake16k')
HF_BAL = os.path.join(ROOT, 'dataset', 'balanced')
DST = os.path.join(ROOT, 'dataset', 'multi')
SR = 22050
TARGET_N = SR * 3
MIN_N = SR * 2

FAKE_SOURCES = {  # name -> (dir, cap segments)
    'conformer_fastspeech2': ('common_voices_prompts_from_conformer_fastspeech2_pwg_ljspeech', 200),
    'full_band_melgan': ('ljspeech_full_band_melgan', 200),
    'hifiGAN': ('ljspeech_hifiGAN', 200),
    'melgan': ('ljspeech_melgan', 200),
    'melgan_large': ('ljspeech_melgan_large', 200),
    'multi_band_melgan': ('ljspeech_multi_band_melgan', 200),
    'parallel_wavegan': ('ljspeech_parallel_wavegan', 200),
    'waveglow': ('ljspeech_waveglow', 200),
    'hf_tts': (None, 186),   # our original HF fakes (kept as 9th source)
}
REAL_SOURCES = {  # name -> (dir, cap segments)
    'ljspeech': (os.path.join(WF, 'none'), 800),
    'hf_real': (os.path.join(HF_BAL, 'real'), 800),
}


def load(p):
    y, sr = sf.read(p, dtype='float32')
    if y.ndim > 1:
        y = y.mean(axis=1)
    if sr != SR:
        n = int(len(y) * SR / sr)
        y = np.interp(np.linspace(0, len(y) - 1, n), np.arange(len(y)), y).astype('float32')
    return y


def slices_3s(y):
    for i in range(0, len(y) - TARGET_N + 1, TARGET_N):
        yield y[i:i + TARGET_N]


def build_stitched(files, cap):
    """Stitch short clips to full 3s segments (for vocoder fakes)."""
    out, buf, buf_len = [], [], 0
    for p in sorted(files):
        try:
            y = load(p)
        except Exception:
            continue
        if len(y) < SR * 0.5:
            continue
        # long clip? slice directly
        if len(y) >= TARGET_N:
            for s in slices_3s(y):
                out.append(s)
                if len(out) >= cap:
                    return out
            continue
        buf.append(y); buf_len += len(y)
        while buf_len >= TARGET_N:
            cat = np.concatenate(buf)
            out.append(cat[:TARGET_N])
            rest = cat[TARGET_N:]
            buf, buf_len = [rest] if len(rest) else [], len(rest)
            if len(out) >= cap:
                return out
    return out


def build_sliced(files, cap):
    """Slice long clips into 3s segments (for real speech)."""
    out = []
    for p in sorted(files):
        try:
            y = load(p)
        except Exception:
            continue
        for s in slices_3s(y):
            out.append(s)
            if len(out) >= cap:
                return out
    return out


def save_seg(cls, idx, seg, outdir):
    m = np.abs(seg).max()
    if m > 1e-6:
        seg = seg / m * 0.95
    sf.write(os.path.join(outdir, f'{cls}_{idx:05d}.wav'), seg, SR)


def main(out_dir=None, fake_mul=1.0, real_mul=1.0):
    global DST
    if out_dir:
        DST = out_dir
    for k in FAKE_SOURCES:
        FAKE_SOURCES[k] = (FAKE_SOURCES[k][0], int(FAKE_SOURCES[k][1] * fake_mul))
    for k in REAL_SOURCES:
        REAL_SOURCES[k] = (REAL_SOURCES[k][0], int(REAL_SOURCES[k][1] * real_mul))
    for cls in ('real', 'fake'):
        os.makedirs(os.path.join(DST, cls), exist_ok=True)
    manifest, idx = {}, {'real': 0, 'fake': 0}

    print('--- FAKE sources (8 vocoders + HF) ---')
    for name, (d, cap) in FAKE_SOURCES.items():
        files = (glob.glob(os.path.join(HF_BAL, 'fake', '*.wav')) if d is None
                 else glob.glob(os.path.join(WF, d, '*.wav')))
        segs = build_stitched(files, cap)
        for s in segs:
            save_seg('fake', idx['fake'], s, os.path.join(DST, 'fake'))
            idx['fake'] += 1
        manifest[f'fake/{name}'] = len(segs)
        print(f'  {name:24s}: {len(segs)} segments')

    print('--- REAL sources ---')
    for name, (d, cap) in REAL_SOURCES.items():
        files = glob.glob(os.path.join(d, '*.wav'))
        segs = build_sliced(files, cap)
        for s in segs:
            save_seg('real', idx['real'], s, os.path.join(DST, 'real'))
            idx['real'] += 1
        manifest[f'real/{name}'] = len(segs)
        print(f'  {name:24s}: {len(segs)} segments')

    manifest['total'] = {'real': idx['real'], 'fake': idx['fake']}
    json.dump(manifest, open(os.path.join(DST, 'sources.json'), 'w'), indent=1)
    print(f"\nDONE: real={idx['real']} fake={idx['fake']} -> dataset/multi/")


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=None)
    ap.add_argument('--fake-mul', type=float, default=1.0)
    ap.add_argument('--real-mul', type=float, default=1.0)
    a = ap.parse_args()
    main(a.out, a.fake_mul, a.real_mul)

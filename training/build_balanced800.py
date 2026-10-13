"""Rebuild duration-balanced 800 real + 800 fake dataset.

Sources (diverse, no single-source dominance):
  REAL 800: 300 multi_v2 real + 300 LibriSpeech slices + 200 MLAAD originals
  FAKE 800: 300 multi_v2 fake (vocoders+HF) + 300 MLAAD TTS slices
            + 100 field ElevenLabs + 100 WaveFake raw vocoder slices
Output: dataset/balanced/{real,fake}/*.wav (all 3.0s @22050Hz) + sources.json
Also restores the evaluate.py same-corpus probe target.
"""
import os, glob, json, random, shutil

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_multisource import load, slices_3s, save_seg

ROOT = r"C:\Users\Kishore GK\Desktop\VIDS"
M2 = os.path.join(ROOT, 'dataset', 'multi_v2')
MLAAD = os.path.join(ROOT, 'dataset', 'hf_mlaad_tiny')
LIBRI = os.path.join(ROOT, 'dataset', 'kaggle', 'librispeech', 'LibriSpeech')
WF = os.path.join(ROOT, 'dataset', 'kaggle', 'wavefake16k')
FIELD = os.path.join(ROOT, 'dataset', 'field')
DST = os.path.join(ROOT, 'dataset', 'balanced')


def copy_some(files, cls, idx, n, seed):
    rnd = random.Random(seed)
    picks = rnd.sample(sorted(files), min(n, len(files)))
    d = os.path.join(DST, cls)
    for p in picks:
        shutil.copy(p, os.path.join(d, f'blr_{idx[0]:05d}.wav'))
        idx[0] += 1
    return len(picks)


def slice_some(files, cls, idx, n, seed):
    rnd = random.Random(seed)
    files = list(files)
    rnd.shuffle(files)
    d = os.path.join(DST, cls)
    got = 0
    for p in files:
        if got >= n:
            break
        try:
            y = load(p)
        except Exception:
            continue
        for s in slices_3s(y):
            if got >= n:
                break
            save_seg(cls, idx[0], s, d)
            idx[0] += 1
            got += 1
    # rename to blr_ prefix for uniformity
    return got


def main():
    for cls in ('real', 'fake'):
        os.makedirs(os.path.join(DST, cls), exist_ok=True)
    manifest, idx = {}, [0]
    parts = {}

    # ---- REAL 800 ----
    r = 0
    r += copy_some(glob.glob(os.path.join(M2, 'real', '*.wav')), 'real', idx, 300, 31)
    parts['real/multi_v2'] = 300
    libri = sorted(glob.glob(os.path.join(LIBRI, '*', '*', '*', '*.flac')))
    got = slice_some(libri, 'real', idx, 300, 32)
    r += got
    parts['real/librispeech'] = got
    orig = sorted(glob.glob(os.path.join(MLAAD, 'original', '*', '*.wav')))
    got = slice_some(orig, 'real', idx, 200, 33)
    r += got
    parts['real/mlaad_orig'] = got

    # ---- FAKE 800 ----
    idx[0] = 0
    f = 0
    f += copy_some(glob.glob(os.path.join(M2, 'fake', '*.wav')), 'fake', idx, 300, 34)
    parts['fake/multi_v2'] = 300
    sys_dirs = sorted(glob.glob(os.path.join(MLAAD, 'fake', '*', '*')))
    mla_files = []
    for sdir in sys_dirs:
        mla_files.extend(sorted(glob.glob(os.path.join(sdir, '*.wav'))))
    got = slice_some(mla_files, 'fake', idx, 300, 35)
    f += got
    parts['fake/mlaad_tts'] = got
    f += copy_some(glob.glob(os.path.join(FIELD, 'fake', '*.wav')), 'fake', idx, 100, 36)
    parts['fake/field_elevenlabs'] = min(100, len(glob.glob(os.path.join(FIELD, 'fake', '*.wav'))))
    wf_dirs = sorted(glob.glob(os.path.join(WF, 'ljspeech_*')))
    wf_files = []
    for wdir in wf_dirs:
        wf_files.extend(sorted(glob.glob(os.path.join(wdir, '*.wav'))))
    got = slice_some(wf_files, 'fake', idx, 100, 37)
    f += got
    parts['fake/wavefake_raw'] = got

    # resequence cleanly to blr_ names (two-pass via tmp_)
    for cls in ('real', 'fake'):
        d = os.path.join(DST, cls)
        for f in os.listdir(d):
            if f.startswith('tmp_'):
                os.remove(os.path.join(d, f))
    for cls in ('real', 'fake'):
        d = os.path.join(DST, cls)
        files = sorted(os.listdir(d))
        for i, fn in enumerate(files):
            os.rename(os.path.join(d, fn), os.path.join(d, f'tmp_{i:05d}.wav'))
        for i, fn in enumerate(sorted(os.listdir(d))):
            os.rename(os.path.join(d, fn), os.path.join(d, f'blr_{i:05d}.wav'))

    manifest['parts'] = parts
    manifest['total'] = {'real': len(os.listdir(os.path.join(DST, 'real'))),
                         'fake': len(os.listdir(os.path.join(DST, 'fake')))}
    json.dump(manifest, open(os.path.join(DST, 'sources.json'), 'w'), indent=1)
    print(f"\nBALANCED800 DONE: real={manifest['total']['real']} "
          f"fake={manifest['total']['fake']} -> dataset/balanced/")


if __name__ == '__main__':
    main()

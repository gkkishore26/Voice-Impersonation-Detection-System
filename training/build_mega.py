"""Build MEGA balanced dataset: the full-training corpus.

Sources (balanced classes, many generators/speakers):
  FAKE: WaveFake 8 vocoders + HF TTS (reuse multi_v2) + MLAAD per-TTS caps
        + field ElevenLabs user uploads
  REAL: LJSpeech + HF real (reuse multi_v2) + LibriSpeech 73 speakers
        + MLAAD originals
Class ratio handled by weighted loss; per-source caps stop dominance.

Output: dataset/mega/{real,fake}/*.wav + sources.json
"""
import os, glob, json, random, shutil

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_multisource import load, slices_3s, save_seg, TARGET_N, SR

ROOT = r"C:\Users\Kishore GK\Desktop\VIDS"
M2 = os.path.join(ROOT, 'dataset', 'multi_v2')
MLAAD = os.path.join(ROOT, 'dataset', 'hf_mlaad_tiny')
LIBRI = os.path.join(ROOT, 'dataset', 'kaggle', 'librispeech', 'LibriSpeech')
FIELD = os.path.join(ROOT, 'dataset', 'field')
DST = os.path.join(ROOT, 'dataset', 'mega')

MLAAD_SYS_CAP = 60     # per TTS system (64 systems -> ~3800)
LIBRI_CAP = 2500       # sliced 3s segments, shuffled speakers
MLAAD_ORIG_CAP = 800


def copy_in(files, cls, idx, manifest, tag):
    d = os.path.join(DST, cls)
    n = 0
    for p in files:
        ext = os.path.splitext(p)[1]
        name = f'{tag}_{idx[cls]:05d}{ext}'
        shutil.copy(p, os.path.join(d, name))
        idx[cls] += 1
        n += 1
    manifest[f'{cls}/{tag}'] = n


def slice_and_save(files, cls, idx, manifest, tag, cap, seed=21):
    d = os.path.join(DST, cls)
    rnd = random.Random(seed)
    files = list(files)
    rnd.shuffle(files)
    n = 0
    for p in files:
        if n >= cap:
            break
        try:
            y = load(p)
        except Exception:
            continue
        for s in slices_3s(y):
            if n >= cap:
                break
            save_seg(cls, 10_000 + idx[cls], s, d)   # offset avoids name clash
            idx[cls] += 1
            n += 1
    manifest[f'{cls}/{tag}'] = n


def main():
    for cls in ('real', 'fake'):
        os.makedirs(os.path.join(DST, cls), exist_ok=True)
    manifest, idx = {}, {'real': 0, 'fake': 0}

    # ---- reuse multi_v2 (8 vocoders + HF + LJSpeech) ----
    copy_in(sorted(glob.glob(os.path.join(M2, 'fake', '*.wav'))),
            'fake', idx, manifest, 'wavefake8_hf')
    copy_in(sorted(glob.glob(os.path.join(M2, 'real', '*.wav'))),
            'real', idx, manifest, 'ljspeech_hfreal')

    # ---- field ElevenLabs (user uploads) ----
    copy_in(sorted(glob.glob(os.path.join(FIELD, 'fake', '*.wav'))),
            'fake', idx, manifest, 'field_elevenlabs')

    # ---- MLAAD fakes: cap per TTS system dir ----
    sys_dirs = sorted(glob.glob(os.path.join(MLAAD, 'fake', '*', '*')))
    rnd = random.Random(22)
    for sdir in sys_dirs:
        wavs = sorted(glob.glob(os.path.join(sdir, '*.wav')))
        if not wavs:
            continue
        tag = 'mla_' + os.path.basename(os.path.dirname(sdir)) + '_' + os.path.basename(sdir)
        tag = ''.join(c if (c.isalnum() or c == '_') else '_' for c in tag)[:48]
        slice_and_save(rnd.sample(wavs, min(len(wavs), MLAAD_SYS_CAP * 3)),
                       'fake', idx, manifest, tag, MLAAD_SYS_CAP, seed=23)

    # ---- MLAAD originals (bonafide, multi-speaker) ----
    orig = sorted(glob.glob(os.path.join(MLAAD, 'original', '*', '*.wav')))
    if orig:
        slice_and_save(orig, 'real', idx, manifest, 'mla_originals', MLAAD_ORIG_CAP)

    # ---- LibriSpeech (73 speakers, clean + noisy) ----
    libri = sorted(glob.glob(os.path.join(LIBRI, '*', '*', '*', '*.flac')))
    if libri:
        slice_and_save(libri, 'real', idx, manifest, 'librispeech', LIBRI_CAP)

    manifest['total'] = {'real': idx['real'], 'fake': idx['fake']}
    json.dump(manifest, open(os.path.join(DST, 'sources.json'), 'w'), indent=1)
    print(f"\nMEGA DONE: real={idx['real']} fake={idx['fake']} -> dataset/mega/")


if __name__ == '__main__':
    main()

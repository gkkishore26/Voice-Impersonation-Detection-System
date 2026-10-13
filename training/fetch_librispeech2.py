"""Download LibriSpeech test-clean + test-other direct from OpenSLR.

73 real speakers (clean + noisy conditions). ~700MB total. No HF needed.
Output: dataset/kaggle/librispeech/{test-clean,test-other}/...
"""
import os, urllib.request, tarfile

OUT = r"C:\Users\Kishore GK\Desktop\VIDS\dataset\kaggle\librispeech"
os.makedirs(OUT, exist_ok=True)

URLS = [
    ('https://www.openslr.org/resources/12/test-clean.tar.gz', 'test-clean.tar.gz'),
    ('https://www.openslr.org/resources/12/test-other.tar.gz', 'test-other.tar.gz'),
]


def show(pct, bs, total, name):
    if int(pct * 100) % 10 == 0:
        print(f'{name}: {pct * 100:.0f}%', flush=True)


for url, fn in URLS:
    dest = os.path.join(OUT, fn)
    if os.path.exists(dest) and os.path.getsize(dest) > 1e6:
        print(f'{fn} already present, skipping download')
    else:
        print(f'downloading {fn}...', flush=True)
        urllib.request.urlretrieve(url, dest,
                                   lambda b, bs, t, n=fn: show(b * bs / t, bs, t, n))
        print(f'{fn} done ({os.path.getsize(dest)/1e6:.0f} MB)', flush=True)
    flag = os.path.join(OUT, fn.replace('.tar.gz', '.extracted'))
    if not os.path.exists(flag):
        print(f'extracting {fn}...', flush=True)
        with tarfile.open(dest) as t:
            t.extractall(OUT)
        open(flag, 'w').write('ok')
        print(f'{fn} extracted', flush=True)

import glob
n = len(glob.glob(os.path.join(OUT, 'LibriSpeech', '*', '*', '*.flac')))
print(f'DONE: {n} flac files -> {OUT}/LibriSpeech/')

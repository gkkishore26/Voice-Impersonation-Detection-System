"""Fetch real Tamil clips (Common Voice ta, validated), capped at 15.

Output: dataset/kaggle/tamil_real/*.mp3 (original clips, untouched)
"""
import os

OUT = r"C:\Users\Kishore GK\Desktop\VIDS\dataset\kaggle\tamil_real"
os.makedirs(OUT, exist_ok=True)
MAX_CLIPS = 15

from datasets import load_dataset, Audio

ds = load_dataset('mozilla-foundation/common_voice_16_1', 'ta',
                  split='validated', streaming=True)
ds = ds.cast_column('audio', Audio(sampling_rate=16000))
n = 0
for ex in ds:
    if n >= MAX_CLIPS:
        break
    a = ex['audio']
    if a is None or a['array'] is None or len(a['array']) < 16000:
        continue
    import soundfile as sf
    sf.write(os.path.join(OUT, f'tamil_real_{n:02d}.wav'), a['array'], 16000)
    n += 1
    print(f'saved tamil_real_{n:02d}.wav', flush=True)
print(f'TAMIL REAL DONE: {n} clips -> {OUT}')

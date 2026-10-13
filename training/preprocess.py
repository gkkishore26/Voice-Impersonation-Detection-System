"""
Preprocessing: Audio -> Mel-Spectrogram image (128 x 128)

Shared by training AND web app so both use identical settings:
    sample rate : 22050 Hz
    duration    : 3 seconds  (66150 samples)
    n_mels      : 128
    n_fft       : 2048
    hop_length  : 512
"""
import os
import numpy as np
import librosa
import librosa.display
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ---------- shared config (single source of truth) ----------
SR = 22050
DURATION = 3
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512
TARGET_SHAPE = (N_MELS, 128)   # 128 mels x ~129 frames -> crop to 128
CLASS_NAMES = ['fake', 'real']  # index 0 = fake (AI), 1 = real (human)


def ffmpeg_load(path, sr=SR, duration=None):
    """Decode ANY audio format (webm/opus, mp3, m4a...) via system ffmpeg.

    Returns mono float32 waveform at `sr`. librosa 1.x dropped its
    audioread fallback, so browser-recorded .webm needs this path.
    """
    import subprocess
    cmd = ['ffmpeg', '-v', 'error', '-i', path,
           '-ac', '1', '-ar', str(sr), '-f', 's16le', '-acodec', 'pcm_s16le', '-']
    raw = subprocess.run(cmd, stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE, check=True).stdout
    y = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if duration is not None:
        y = y[:int(sr * duration)]
    return y


def load_audio(path, sr=SR, duration=DURATION):
    """Load audio, mono, fixed length (pad or trim)."""
    try:
        y, _ = librosa.load(path, sr=sr, mono=True, duration=duration)
    except Exception:
        y = ffmpeg_load(path, sr, duration)   # e.g. browser .webm recordings
    target = sr * duration
    if len(y) < target:
        y = np.pad(y, (0, target - len(y)))      # pad with silence
    else:
        y = y[:target]                           # trim
    return y


def mel_spectrogram(y, sr=SR):
    """Waveform -> log-mel-spectrogram (numpy array, 128 x 128)."""
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=N_FFT,
                                         hop_length=HOP_LENGTH, n_mels=N_MELS)
    log_mel = librosa.power_to_db(mel, ref=np.max)   # log scale, ref=max (0 dB)
    # crop / pad frames to width 128
    if log_mel.shape[1] >= 128:
        log_mel = log_mel[:, :128]
    else:
        log_mel = np.pad(log_mel, ((0, 0), (0, 128 - log_mel.shape[1])))
    return log_mel.astype(np.float32)


def segment_to_input(y):
    """Waveform (any length) -> model input (1,128,128), normalised to [0,1]."""
    y = np.asarray(y, dtype=np.float32)
    y = y - np.mean(y)                       # DC block (recording offset)
    peak = np.abs(y).max()
    if peak > 1e-6:
        y = y / peak * 0.95                  # peak normalize (matches training build)
    # NOTE: mel uses power_to_db(ref=max), so level cannot bias the model.
    # No spectral denoising: the CNN was trained on raw spectra; filtering
    # would shift the distribution. Noise is reported, not removed.
    target = SR * DURATION
    if len(y) < target:
        y = np.pad(y, (0, target - len(y)))      # pad silence
    else:
        y = y[:target]                           # trim
    mel = mel_spectrogram(y)
    mn, mx = float(mel.min()), float(mel.max())
    if mx - mn > 1e-6:
        mel = (mel - mn) / (mx - mn)
    return mel[np.newaxis, :, :].astype(np.float32)   # (1, 128, 128)


def audio_to_input(path):
    """Full pipeline: file path -> model input array (1,128,128)."""
    y = load_audio(path)
    return segment_to_input(y)


def audio_to_chunks(path, sr=SR, chunk_sec=DURATION, max_sec=300):
    """
    Split a LONG audio file into chunk_sec segments.
    Returns: list of (start_seconds, (1,128,128) input)
    Last chunk is kept only if >= half length (avoids tiny tail).
    """
    y = None
    try:
        y, _ = librosa.load(path, sr=sr, mono=True, duration=max_sec)
    except Exception:
        y = ffmpeg_load(path, sr, max_sec)   # e.g. browser .webm recordings
    n = int(sr * chunk_sec)
    items = []
    for i in range(0, len(y), n):
        seg = y[i:i + n]
        if len(seg) < n // 2:
            break                                 # skip tiny tail
        items.append((i / sr, segment_to_input(seg)))
    return items


def save_mel_image(mel, out_path, title=None, duration=3.0):
    """Save a spectrogram as a wide labeled PNG (full graph for web/report)."""
    fig, ax = plt.subplots(figsize=(8.0, 3.0))
    ax.imshow(mel, aspect='auto', origin='lower', cmap='magma')
    # full-graph axes: time (s) x mel bands
    ticks = [0, 32, 64, 96, 128]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f'{duration * i / 128:.1f}' for i in ticks], fontsize=7)
    ax.set_yticks(ticks)
    ax.set_yticklabels([str(i) for i in ticks], fontsize=7)
    ax.set_xlabel(f'Time (seconds, 0-{duration:.1f} s window)', fontsize=8)
    ax.set_ylabel('Mel bands (0-128)', fontsize=8)
    if title:
        ax.set_title(title, fontsize=9)
    fig.tight_layout(pad=0.4)
    fig.savefig(out_path, dpi=110, bbox_inches='tight', facecolor='white')
    plt.close(fig)


if __name__ == '__main__':
    import sys, glob
    # self-test: build spectrograms from a few downloaded samples
    root = r"C:\Users\Kishore GK\Desktop\VIDS\dataset\raw"
    outdir = r"C:\Users\Kishore GK\Desktop\VIDS\dataset\_samples"
    os.makedirs(outdir, exist_ok=True)
    got = 0
    for cls in ('real', 'fake'):
        files = glob.glob(os.path.join(root, cls, '*'))
        files = [f for f in files if not f.endswith('.DS_Store')]
        for f in files[:2]:
            mel = audio_to_input(f)
            print(f'{cls}/{os.path.basename(f)} -> shape {mel.shape}, '
                  f'range [{mel.min():.3f}, {mel.max():.3f}]')
            save_mel_image(mel[0], os.path.join(outdir, f'{cls}_{os.path.basename(f)}.png'),
                           title=f'{cls.upper()} sample')
            got += 1
    if got == 0:
        print('No samples yet (dataset still downloading) - testing with synthetic audio')
        y = np.sin(2 * np.pi * 220 * np.arange(SR * 3) / SR).astype(np.float32)
        mel = mel_spectrogram(y)
        print(f'synthetic -> shape {mel.shape}')
    print('SELF-TEST OK')

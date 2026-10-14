"""Rewrite the v1.0 GitHub release body with the full Quick Start guide."""
import json
import subprocess
import urllib.request

T = subprocess.run(
    ['powershell', '-noprofile', '-command',
     "[System.Environment]::GetEnvironmentVariable('GITHUB_PAT','User')"],
    capture_output=True, text=True).stdout.strip()

BODY = """# VIDS v1.0 — Voice Impersonation Detection System

Detect whether a voice clip is a **real human** or **AI-generated** — running
**100% offline** on your own PC. No account, no uploads to any cloud.

## What it does

- **Upload** a voice file (WAV / MP3 / M4A / OGG / FLAC / WEBM, 3 sec – 5 min)
  or **record live** from your microphone (up to 60 seconds).
- Press **Analyze Voice** and Professor K — a 4.58M-parameter convolutional
  neural network reading mel-spectrograms — scores every 3-second block and
  combines the votes into one verdict.
- Each report shows the **verdict + confidence**, an **acoustic-evidence panel**
  (pitch variation, spectral markers, loudness dynamics), the **spectrogram
  graph**, a per-block **segment map** for long clips, and a **print-ready**
  report button.
- Borderline cases return an explicit **UNCERTAIN verdict** instead of a
  forced guess.

## Quick Start

1. Download **Setup.exe** below and run it (no administrator rights needed;
   installs into your user folder).
2. Launch **VIDS** from the Start Menu (a desktop shortcut is optional).
3. First launch takes about a minute (engines initialize); afterwards it is fast.
4. Upload a clip or record your voice, click **Analyze Voice**, read the report.
5. Allow microphone access when the browser-style prompt appears for recording.

## How it was built

- **Model**: custom 4-block CNN (32→64→128→256 filters, batch norm, dropout),
  input 128×128 log-mel spectrogram (22.05 kHz, 3 s windows), PyTorch.
- **Training data**: 15,691 segments — WaveFake 8 vocoders, MLAAD systems,
  LibriSpeech speakers, plus user-provided labelled field recordings.
  Augmentation: time-shift, noise, gain, SpecAugment; class-weighted loss;
  temperature-calibrated (T=2.5) confidence.
- **App**: Flask backend + desktop window (pywebview), packaged with
  PyInstaller + Inno Setup. Inference runs on CPU, so any PC works.

## Measured performance (verified, not estimated)

| Evaluation | Result |
|---|---|
| Validation, diverse 3,138 segments | **94.6%** acc, F1 0.930, **ROC-AUC 0.9896** |
| Balanced probe, 1,600 segments | **94.4%** acc, F1 0.945 |
| Held-out whole files (never trained on) | **16/16 correct (100%)** |
| Real-voice re-tests | 4/4 correct (68–94% confidence) |
| Multilingual spot-check (Tamil) | 9/11, zero misclassifications |

Evaluation used file-grouped splits (no segment leakage between train and
validation) plus strictly held-out whole files. Validation shares data sources
with training, so treat it as optimistic; the held-out file tests are the
honest grade.

## Limitations

- Accuracy drops on recording conditions absent from training (new languages,
  unusual microphones/rooms); the app reports UNCERTAIN instead of guessing.
- Very short clips (< 3 s) are rejected; very noisy clips get lower confidence.
- Desktop build runs on CPU (GPU build is used only for training).

## System requirements

- Windows 10/11, 64-bit, ~4 GB RAM, ~2 GB free disk, microphone for recording.
- Microsoft Edge WebView2 (preinstalled on Windows 10/11).

## How it compares

- **Cloud detector websites**: require uploading your voice to someone else's
  server and an account; VIDS runs fully offline — recordings never leave PC.
- **Research baselines (e.g. ASVspoof systems)**: strong published numbers on
  benchmarks, but lab setups, not downloadable end-user apps. VIDS is an
  applied, install-and-run system; no claim of beating benchmark SOTA is made.
- **Genuine advantages**: free, private/offline, uncertainty-aware verdicts,
  per-block evidence with spectrogram graphs, printable reports.
"""

req = urllib.request.Request(
    'https://api.github.com/repos/gkkishore26/Voice-Impersonation-Detection-System/releases/tags/v1.0',
    headers={'Authorization': 'Bearer ' + T,
             'Accept': 'application/vnd.github+json'})
rel = json.load(urllib.request.urlopen(req, timeout=30))
body = json.dumps({'body': BODY}).encode()
patch = urllib.request.Request(
    f"https://api.github.com/repos/gkkishore26/Voice-Impersonation-Detection-System/releases/{rel['id']}",
    data=body, method='PATCH',
    headers={'Authorization': 'Bearer ' + T,
             'Accept': 'application/vnd.github+json',
             'Content-Type': 'application/json'})
out = json.load(urllib.request.urlopen(patch, timeout=30))
print('release body updated:', out['tag_name'], '| assets:',
      [(a['name'], a['size']) for a in out.get('assets', [])])

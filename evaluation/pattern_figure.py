"""Recreate 'What the CNN Learns' figure from OUR OWN data.

Left: real human mel-spectrogram (natural variation, full frequency).
Right: AI-generated mel-spectrogram (smoother, regular, missing detail).
Saves evaluation/cnn_pattern_difference.png (report figure).
"""
import os
import glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import sys
sys.path.insert(0, r"C:\Users\Kishore GK\Desktop\VIDS\training")
from preprocess import audio_to_input, mel_spectrogram, load_audio, SR, DURATION

ROOT = r"C:\Users\Kishore GK\Desktop\VIDS"
MULTI = os.path.join(ROOT, 'dataset', 'multi')

real_f = sorted(glob.glob(os.path.join(MULTI, 'real', '*.wav')))[0]
fake_f = sorted(glob.glob(os.path.join(MULTI, 'fake', '*.wav')))[0]

# full 3s waveforms -> mel (use raw mel, not normalized input, for display)
mel_r = mel_spectrogram(load_audio(real_f))
mel_f = mel_spectrogram(load_audio(fake_f))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2))
fig.suptitle('What the CNN Learns: Mel-Spectrogram Pattern Difference',
             fontsize=15, fontweight='bold')

for ax, mel, title, color, caption in [
    (ax1, mel_r, 'Real Human Voice', '#6aa84f',
     'Natural variation  +  fine detail  +  full frequency'),
    (ax2, mel_f, 'AI-Generated Voice', '#cc0000',
     'Too smooth  +  regular pattern  +  missing details'),
]:
    ax.imshow(mel, aspect='auto', origin='lower', cmap='magma')
    ax.set_title(title, fontsize=14, fontweight='bold', color=color, pad=10)
    ax.set_xlabel('Time →', fontsize=11)
    ax.set_ylabel('Frequency →', fontsize=11)
    ax.text(0.5, -0.22, caption, transform=ax.transAxes, ha='center',
            fontsize=11, color='#444444')
    for spine in ax.spines.values():
        spine.set_edgecolor(color)
        spine.set_linewidth(3)

fig.tight_layout(rect=[0, 0.06, 1, 0.93])
out = os.path.join(ROOT, 'evaluation', 'cnn_pattern_difference.png')
fig.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
plt.close(fig)
print('saved', out)
print('real:', os.path.basename(real_f), '| fake:', os.path.basename(fake_f))

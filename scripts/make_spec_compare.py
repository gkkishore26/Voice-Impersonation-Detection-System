# Fig 5.3: Real Human Voice vs AI-Generated Voice Mel-Spectrogram comparison
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

OUT = r"C:\Users\Kishore GK\Desktop\VIDS\Report\figures"
rng = np.random.default_rng(42)

W, H = 120, 100
t = np.linspace(0, 1, W)
f = np.linspace(0, 1, H)

# ---- Real human voice: harmonic bands with jitter + broadband noise detail ----
real = np.zeros((H, W))
for k in range(1, 14):  # harmonic formant bands
    base = 0.06 + 0.07 * k ** 0.75
    jitter = 0.012 * np.sin(2 * np.pi * (1.5 + 0.4 * k) * t + k)  # pitch variation
    vib = 0.010 * rng.standard_normal(W).cumsum() / W  # natural micro-vibrato
    for y in range(H):
        center = base + jitter + 0.02 * np.sin(2 * np.pi * 0.8 * t)
        d = np.abs(f[y] - center)
        real[y] += np.exp(-(d ** 2) / (2 * 0.010 ** 2)) * (1.0 / (1 + 0.25 * k))
real += 0.55 * rng.random((H, W))          # breath / air / room noise detail
real += 0.30 * np.exp(-((f[:, None] - 0.55 - 0.25 * np.sin(2 * np.pi * t))[None, :] if False else 0)) if False else 0
# energy emphasis in lower-mid frequencies (natural speech energy)
real *= np.exp(-((f[:, None] - 0.35) ** 2) / 0.18) + 0.35
real = (real - real.min()) / (real.max() - real.min())

# ---- AI-generated voice: overly smooth, regular, missing high-frequency ----
fake = np.zeros((H, W))
for k in range(1, 11):                    # fewer, perfectly regular harmonics
    base = 0.07 + 0.075 * k ** 0.78
    for y in range(H):
        d = np.abs(f[y] - base)
        fake[y] += np.exp(-(d ** 2) / (2 * 0.014 ** 2)) * (1.0 / (1 + 0.22 * k))
fake *= 1.0 + 0.04 * np.sin(2 * np.pi * 0.5 * t)   # unnaturally steady amplitude
fake[H // 3:, :] *= 0.10                  # high frequencies almost missing
fake += 0.04 * rng.random((H, W))         # very little natural noise
fake = (fake - fake.min()) / (fake.max() - fake.min())

fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.6))
for ax, img, title, sub, edge in (
    (axes[0], real, 'Real Human Voice', 'Natural variation  +  fine detail  +  full frequency', '#70AD47'),
    (axes[1], fake, 'AI-Generated Voice', 'Too smooth  +  regular pattern  +  missing details', '#C00000'),
):
    im = ax.imshow(img, aspect='auto', origin='lower', cmap='magma', interpolation='bilinear')
    ax.set_title(title, fontsize=12.5, fontweight='bold', color=edge, pad=10)
    ax.text(0.5, -0.16, sub, transform=ax.transAxes, ha='center', fontsize=9.5, color='#333333')
    ax.set_xlabel('Time  →', fontsize=9.5)
    ax.set_ylabel('Frequency  →', fontsize=9.5)
    for s in ax.spines.values():
        s.set_color(edge)
        s.set_linewidth(2)

fig.suptitle('What the CNN Learns: Mel-Spectrogram Pattern Difference',
             fontsize=13, fontweight='bold', y=1.02)
fig.tight_layout()
fig.savefig(OUT + r'\fig9_spectrogram_compare.png', dpi=200, bbox_inches='tight', facecolor='white')
print('saved fig9_spectrogram_compare.png')

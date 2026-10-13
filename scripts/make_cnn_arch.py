# CNN architecture diagram (Fig 5.2)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = r"C:\Users\Kishore GK\Desktop\VIDS\Report\figures"

BLUE = '#4F81BD'; LBLUE = '#D9E4F5'; ORANGE = '#F79646'; LORANGE = '#FDE9D9'
GREEN = '#70AD47'; LGREEN = '#E2EFDA'; GRAY = '#D9D9D9'; INK = '#222222'

fig, ax = plt.subplots(figsize=(13, 5.2))
ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis('off')

blocks = [
    ('Input\n128x128\nx1', GRAY, '#808080', False),
    ('Conv2D\n32\nBN + ReLU', LBLUE, BLUE, False),
    ('MaxPool\n2x2', LBLUE, BLUE, False),
    ('Conv2D\n64\nBN + ReLU', LBLUE, BLUE, False),
    ('MaxPool\n2x2', LBLUE, BLUE, False),
    ('Conv2D\n128\nBN + ReLU', LBLUE, BLUE, False),
    ('MaxPool\n2x2', LBLUE, BLUE, False),
    ('Conv2D\n256\nBN + ReLU', LBLUE, BLUE, False),
    ('MaxPool\n2x2', LBLUE, BLUE, False),
    ('Flatten', LORANGE, ORANGE, False),
    ('Dropout\n0.5', LORANGE, ORANGE, False),
    ('Dense\n256\nReLU', LORANGE, ORANGE, False),
    ('Softmax\nDense 2', LGREEN, GREEN, False),
    ('Output\nReal / AI', LGREEN, GREEN, True),
]

n = len(blocks)
x0, w, gap = 1.5, 6.0, 0.85
y, h = 34, 30
for i, (txt, fc, ec, bold) in enumerate(blocks):
    x = x0 + i * (w + gap)
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.2,rounding_size=1.0",
                                fc=fc, ec=ec, lw=1.5))
    ax.text(x + w / 2, y + h / 2, txt, ha='center', va='center', fontsize=7.2,
            fontweight='bold' if bold else 'normal', color=INK, linespacing=1.3)
    if i < n - 1:
        ax.add_patch(FancyArrowPatch((x + w, y + h / 2), (x + w + gap, y + h / 2),
                                     arrowstyle='-|>', mutation_scale=12, lw=1.2, color='#404040',
                                     shrinkA=1, shrinkB=1))

feat_end = x0 + 9 * (w + gap) - gap
cls_start = x0 + 9 * (w + gap)
ax.plot([x0, feat_end], [28, 28], color='#1F4E79', lw=2)
ax.text((x0 + feat_end) / 2, 22, 'Feature Extraction Blocks (Convolutional Layers)',
        ha='center', fontsize=9, color='#1F4E79', fontweight='bold')
ax.plot([cls_start, x0 + n * (w + gap) - gap], [28, 28], color='#C55A11', lw=2)
ax.text((cls_start + x0 + n * (w + gap) - gap) / 2, 22, 'Classification Head (Dense Layers)',
        ha='center', fontsize=9, color='#C55A11', fontweight='bold')
ax.text(50, 8, 'Input: Mel-spectrogram image (grayscale, 128x128)   |   Output: Softmax probabilities [P(Real), P(AI-Generated)]',
        ha='center', fontsize=8.5, color='#404040', style='italic')

fig.savefig(OUT + r'\fig3_cnn_architecture.png', dpi=200, bbox_inches='tight', facecolor='white')
print('saved fig3_cnn_architecture.png')

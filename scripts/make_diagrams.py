# Generates all report diagrams for the Voice Impersonation Detection System project
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Ellipse, Polygon, Circle, FancyArrowPatch, Rectangle
import os

OUT = r"C:\Users\Kishore GK\Desktop\VIDS\Report\figures"
os.makedirs(OUT, exist_ok=True)

BLUE = '#4F81BD'; LBLUE = '#D9E4F5'; DBLUE = '#1F4E79'
ORANGE = '#F79646'; LORANGE = '#FDE9D9'
GREEN = '#70AD47'; LGREEN = '#E2EFDA'
GRAY = '#D9D9D9'; YEL = '#FFF2CC'; DYEL = '#BF9000'
INK = '#222222'; ARR = '#404040'


def newfig(w, h):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis('off')
    return fig, ax


def rbox(ax, x, y, w, h, text, fc=LBLUE, ec=BLUE, fs=9, weight='normal', lw=1.5, tc=INK):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.2,rounding_size=1.2",
                                fc=fc, ec=ec, lw=lw))
    ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=fs,
            fontweight=weight, color=tc, linespacing=1.35)


def sqbox(ax, x, y, w, h, text, fc=LBLUE, ec=BLUE, fs=9, weight='normal', lw=1.5, tc=INK):
    ax.add_patch(Rectangle((x, y), w, h, fc=fc, ec=ec, lw=lw))
    ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=fs,
            fontweight=weight, color=tc, linespacing=1.35)


def arw(ax, p1, p2, label=None, fs=7.8, color=ARR, lw=1.5, rad=0.0, lx=None, ly=None):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle='-|>', mutation_scale=15, lw=lw,
                                 color=color, connectionstyle=f"arc3,rad={rad}",
                                 shrinkA=3, shrinkB=3))
    if label:
        mx = lx if lx is not None else (p1[0] + p2[0]) / 2
        my = ly if ly is not None else (p1[1] + p2[1]) / 2
        ax.text(mx, my, label, ha='center', va='bottom', fontsize=fs, color=color, linespacing=1.2)


def diamond(ax, cx, cy, w, h, text, fc=YEL, ec=DYEL, fs=8.5):
    ax.add_patch(Polygon([(cx, cy + h/2), (cx + w/2, cy), (cx, cy - h/2), (cx - w/2, cy)],
                         closed=True, fc=fc, ec=ec, lw=1.5))
    ax.text(cx, cy, text, ha='center', va='center', fontsize=fs, color=INK, linespacing=1.3)


def save(fig, name):
    fig.savefig(os.path.join(OUT, name), dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print('saved', name)


# ---------------- Fig 4.1 System Architecture ----------------
def fig1():
    fig, ax = newfig(8.5, 7.5)
    sqbox(ax, 28, 84, 44, 8, 'User (Student / Voice Analyst)', fc=YEL, ec=DYEL, fs=10.5, weight='bold')
    rbox(ax, 16, 66, 68, 12,
         'Presentation Layer — Web Interface (HTML / CSS / JavaScript)\n'
         'Upload Audio File   •   Live Mic Recording (MediaRecorder API)\n'
         'Result Display with Confidence Gauge', fs=8.5)
    rbox(ax, 12, 46, 76, 14,
         'Application Layer — Flask Backend (Python)\n'
         '• HTTP Request Handling (POST /predict)\n'
         '• Audio Format & Duration Validation\n'
         '• Audio → Mel-Spectrogram Conversion (librosa)', fs=8.5)
    rbox(ax, 12, 26, 76, 14,
         'Machine Learning Layer — Trained CNN Model (PyTorch)\n'
         '• Convolutional Feature Extraction (4 Conv Blocks)\n'
         '• Dense Classification Head\n'
         '• Softmax: Probability of Real vs AI-Generated', fs=8.5)
    rbox(ax, 22, 10, 56, 8, 'Storage Layer — Trained Model File (voice_cnn.pth)',
         fc=LGREEN, ec=GREEN, fs=9)
    arw(ax, (50, 84), (50, 78.5))
    arw(ax, (50, 66), (50, 60.5))
    arw(ax, (50, 46), (50, 40.5))
    arw(ax, (50, 26), (50, 18.5))
    arw(ax, (8, 33), (8, 53), 'Prediction\nProbability', fs=7.5, lx=5.5, ly=43)
    arw(ax, (8, 53), (8, 72), 'JSON\nResponse', fs=7.5, lx=5.5, ly=62)
    arw(ax, (8, 72), (8, 87), 'Verdict +\nConfidence', fs=7.5, lx=5.5, ly=78)
    save(fig, 'fig1_system_architecture.png')


# ---------------- Fig 4.2 Work Flow ----------------
def fig2():
    fig, ax = newfig(8.5, 10.5)
    rbox(ax, 36, 92, 28, 6, 'Start', fc=LGREEN, ec=GREEN, fs=10, weight='bold')
    rbox(ax, 18, 79, 64, 7.5,
         'User uploads an audio file OR records live voice\n(MediaRecorder API in the browser)')
    rbox(ax, 18, 67, 64, 7.5, 'Browser sends audio to Flask backend  →  POST /predict')
    diamond(ax, 50, 57.5, 36, 11, 'Audio valid?\n(.wav / .mp3, 3–15 s)')
    rbox(ax, 1, 54, 20, 7, 'Display Error\nMessage', fc=YEL, ec=DYEL, fs=8)
    rbox(ax, 18, 39, 64, 7.5, 'Convert audio → Mel-Spectrogram image (128 × 128) using librosa')
    rbox(ax, 18, 27, 64, 7.5, 'CNN model computes class probabilities (Real vs AI-Generated)')
    rbox(ax, 18, 15, 64, 7.5, 'Generate detection report  (predicted label + confidence %)')
    rbox(ax, 18, 3, 64, 6, 'Display result on web interface   →   END',
         fc=LGREEN, ec=GREEN, fs=9.5, weight='bold')
    arw(ax, (50, 92), (50, 86.8))
    arw(ax, (50, 79), (50, 74.8))
    arw(ax, (50, 67), (50, 63.2))
    arw(ax, (32, 57.5), (21.5, 57.5), 'No', fs=8.5, ly=58.3)
    arw(ax, (11, 61.2), (11, 82.5), 'Retry', fs=7.5, lx=7, ly=71)
    arw(ax, (11, 82.5), (18, 82.5))
    arw(ax, (50, 52), (50, 47.2), 'Yes', fs=8.5, lx=54, ly=49)
    arw(ax, (50, 39), (50, 35.2))
    arw(ax, (50, 27), (50, 23.2))
    arw(ax, (50, 15), (50, 9.5))
    save(fig, 'fig2_workflow.png')


# ---------------- Fig 4.3 Use Case Diagram ----------------
def fig4():
    fig, ax = newfig(10.5, 7.5)
    ax.add_patch(Rectangle((22, 8), 75, 84, fill=False, ec=DBLUE, lw=2))
    ax.text(59.5, 95, 'Voice Impersonation Detection System', ha='center',
            fontsize=10.5, fontweight='bold', color=DBLUE)
    # actor
    ax.add_patch(Circle((9, 74), 2.3, fc='white', ec=INK, lw=1.6))
    ax.plot([9, 9], [71.7, 64], color=INK, lw=1.6)
    ax.plot([5.8, 12.2], [68.5, 68.5], color=INK, lw=1.6)
    ax.plot([9, 6.2], [64, 58.5], color=INK, lw=1.6)
    ax.plot([9, 11.8], [64, 58.5], color=INK, lw=1.6)
    ax.text(9, 55, 'User', ha='center', fontsize=9.5, fontweight='bold')
    cases = [('Upload Audio File', 84), ('Record Live Audio', 70), ('Submit for Analysis', 56),
             ('View Detection Report', 42), ('Perform New Test', 28)]
    for txt, cy in cases:
        ax.add_patch(Ellipse((44, cy), 32, 9, fc=LBLUE, ec=BLUE, lw=1.4))
        ax.text(44, cy, txt, ha='center', va='center', fontsize=8)
        ax.plot([11.5, 28], [68 - (68 - cy) * 0.12, cy], color=INK, lw=1.2)
    internal = [('Validate Audio', 70), ('Generate Mel-\nSpectrogram', 52), ('Classify Voice\n(CNN Model)', 34)]
    for txt, cy in internal:
        ax.add_patch(Ellipse((77, cy), 28, 11, fc=LORANGE, ec=ORANGE, lw=1.4, linestyle='dashed'))
        ax.text(77, cy, txt, ha='center', va='center', fontsize=7.8)
        ax.annotate('', xy=(63.2, cy), xytext=(59.8, 56),
                    arrowprops=dict(arrowstyle='-|>', color=ORANGE, lw=1.2, linestyle='dashed'))
        ax.text(61.5, (cy + 56) / 2 + 1.2, '<<include>>', fontsize=6.5, color=ORANGE, ha='center')
    save(fig, 'fig3_usecase.png')


# ---------------- Fig 4.4 DFD Level 0 ----------------
def fig5():
    fig, ax = newfig(9.5, 5.5)
    sqbox(ax, 2, 40, 22, 15, 'User\n(External Entity)', fc=YEL, ec=DYEL, fs=9)
    ax.add_patch(Circle((52, 47.5), 14, fc=LBLUE, ec=BLUE, lw=1.6))
    ax.text(52, 47.5, 'Voice Impersonation\nDetection System\n(Process 0.0)',
            ha='center', va='center', fontsize=8.2, linespacing=1.3)
    sqbox(ax, 78, 40, 20, 15, 'D1: Trained CNN\nModel File\n(voice_cnn.pth)', fc=LGREEN, ec=GREEN, fs=8)
    arw(ax, (24, 52.5), (37.2, 51), 'Audio Input\n(File / Live Mic)', fs=7.5, ly=52.5)
    arw(ax, (37.2, 43.5), (24, 44.5), 'Detection Report\n(Real / AI + Confidence)', fs=7.5, ly=41.5)
    arw(ax, (66.3, 52), (78, 52), 'Load Model & Weights', fs=7.5, ly=52.8)
    arw(ax, (78, 43.5), (66.3, 43.5), 'Prediction Result', fs=7.5, ly=44.3)
    save(fig, 'fig4_dfd_level0.png')


# ---------------- Fig 4.5 DFD Level 1 ----------------
def fig6():
    fig, ax = newfig(10.5, 7.5)
    sqbox(ax, 1, 43, 15, 12, 'User', fc=YEL, ec=DYEL, fs=9.5, weight='bold')
    procs = [(78, '1.0\nManage\nAudio Input'), (58, '2.0\nPreprocess\nAudio'),
             (36, '3.0\nClassify\nVoice (CNN)'), (16, '4.0\nGenerate\nReport')]
    for cy, txt in procs:
        ax.add_patch(Circle((32, cy), 7.6, fc=LBLUE, ec=BLUE, lw=1.5))
        ax.text(32, cy, txt, ha='center', va='center', fontsize=6.9, linespacing=1.25)
    sqbox(ax, 63, 74, 33, 8, 'D1: Audio Buffer', fc=LGREEN, ec=GREEN, fs=8.5)
    sqbox(ax, 63, 32, 33, 8, 'D2: Trained CNN Model', fc=LGREEN, ec=GREEN, fs=8.5)
    sqbox(ax, 63, 12, 33, 8, 'D3: Report Log', fc=LGREEN, ec=GREEN, fs=8.5)
    arw(ax, (16, 51), (25, 73), 'Audio input', fs=7.2)
    arw(ax, (39.7, 78), (63, 78), 'Store raw audio', fs=7.2, ly=78.8)
    arw(ax, (68, 74), (39.5, 60.5), 'Valid audio clip', fs=7.2, lx=56, ly=69)
    arw(ax, (32, 50.4), (32, 43.8), 'Mel-spectrogram', fs=7.2, lx=34.5, ly=45.5)
    arw(ax, (63, 36), (39.8, 36), 'Model parameters', fs=7.2, ly=36.8)
    arw(ax, (32, 28.4), (32, 23.8), 'Class probabilities', fs=7.2, lx=34.5, ly=24.5)
    arw(ax, (24.7, 14), (8.5, 43), 'Detection report', fs=7.2, lx=15, ly=26)
    arw(ax, (39.7, 16), (63, 16), 'Save report log', fs=7.2, ly=16.8)
    save(fig, 'fig5_dfd_level1.png')


# ---------------- Fig 4.6 Sequence Diagram ----------------
def fig7():
    fig, ax = newfig(11.5, 6.8)
    xs = [11, 37, 63, 89]
    names = ['User', 'Web Interface\n(HTML / JS)', 'Flask Backend\n(Python)', 'CNN Model\n(PyTorch)']
    for x, n in zip(xs, names):
        sqbox(ax, x - 11, 90, 22, 6.5, n, fc=LBLUE, fs=8.5, weight='bold')
        ax.plot([x, x], [90, 5], color='#808080', lw=1.2, linestyle='dashed')
    msgs = [
        (0, 1, '1. Upload audio / start live recording', 81, False),
        (1, 2, '2. POST /predict (audio file)', 72.5, False),
        (2, 2, '3. Validate audio format & duration', 64, True),
        (2, 2, '4. Convert audio to mel-spectrogram (librosa)', 55.5, True),
        (2, 3, '5. Input tensor (1 × 1 × 128 × 128)', 47, False),
        (3, 2, '6. Softmax probabilities [P(real), P(fake)]', 38.5, False),
        (2, 1, '7. JSON response { label, confidence }', 30, False),
        (1, 0, '8. Display verdict + confidence gauge', 21.5, False),
    ]
    for a, b, txt, y, self_ in msgs:
        if self_:
            x = xs[a]
            ax.add_patch(FancyArrowPatch((x, y + 1.5), (x + 8, y - 1.5), arrowstyle='-|>',
                                         mutation_scale=13, lw=1.3, color=ARR,
                                         connectionstyle='arc3,rad=-0.6'))
            ax.text(x + 10, y - 2, txt, ha='left', va='center', fontsize=7.8)
        else:
            ax.add_patch(FancyArrowPatch((xs[a], y), (xs[b], y), arrowstyle='-|>',
                                         mutation_scale=14, lw=1.4, color=ARR, shrinkA=1, shrinkB=1))
            ax.text((xs[a] + xs[b]) / 2, y + 1.5, txt, ha='center', fontsize=7.8)
    save(fig, 'fig6_sequence.png')


# ---------------- Fig 4.7 Gantt Chart ----------------
def fig9():
    tasks = [
        ('Requirement Analysis & Literature Survey', 1, 2),
        ('Dataset Collection & Labeling', 2, 2),
        ('Data Preprocessing (Mel-Spectrogram Extraction)', 3, 2),
        ('CNN Model Design & Development', 4, 2),
        ('Model Training & Evaluation', 5, 3),
        ('Web Application Development', 7, 3),
        ('Testing & Debugging', 9, 2),
        ('Documentation & Report Writing', 10, 2),
        ('Presentation & Final Submission', 11, 2),
    ]
    fig, ax = plt.subplots(figsize=(9.8, 5.2))
    colors = [BLUE, '#5B9BD5', '#70AD47', '#FFC000', '#ED7D31', '#4472C4', '#A5A5A5', '#264478', '#C00000']
    for i, (name, start, dur) in enumerate(tasks):
        ax.barh(i, dur, left=start - 0.5, height=0.6, color=colors[i], edgecolor='white')
    ax.set_yticks(range(len(tasks)))
    ax.set_yticklabels([t[0] for t in tasks], fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels([f'W{i}' for i in range(1, 13)], fontsize=8.5)
    ax.set_xlim(0.3, 12.7)
    ax.set_xlabel('Project Weeks', fontsize=9.5)
    ax.grid(axis='x', linestyle='--', alpha=0.5)
    ax.set_facecolor('white')
    fig.tight_layout()
    save(fig, 'fig7_gantt.png')


# ---------------- Fig 5.1 Mel-Spectrogram Pipeline ----------------
def fig8():
    fig, ax = newfig(12.5, 3.8)
    texts = ['Raw Audio\nWaveform\n(.wav / .mp3)', 'Resample\n22,050 Hz\nTrim / Pad 3 s',
             'Pre-emphasis\nFilter', 'STFT\n(n_fft = 2048\nhop = 512)',
             'Mel Filterbank\n128 Mel\nBands', 'Log\nCompression',
             'Mel-Spectrogram\nImage 128 × 128']
    fcs = [GRAY, LBLUE, LBLUE, LBLUE, LBLUE, LBLUE, LGREEN]
    ecs = ['#808080', BLUE, BLUE, BLUE, BLUE, BLUE, GREEN]
    w, step = 11.5, 13.9
    for i, (t, fc, ec) in enumerate(zip(texts, fcs, ecs)):
        x = 2 + i * step
        rbox(ax, x, 28, w, 30, t, fc=fc, ec=ec, fs=7.6,
             weight='bold' if i == 6 else 'normal')
        if i < 6:
            arw(ax, (x + w, 43), (x + step, 43))
    save(fig, 'fig8_melspec_pipeline.png')


fig1(); fig2(); fig4(); fig5(); fig6(); fig7(); fig9(); fig8()
print('ALL DIAGRAMS DONE')

"""Acoustic evidence engine for VIDS.

Layers a transparent, rule-based forensic examination UNDER the CNN verdict:
  1. validate_audio  - reject silence / unvoiced / corrupt files (with reason)
  2. extract_evidence - pitch (yin), spectral stats, MFCCs, dynamics
  3. flag rules       - documented thresholds, each finding tagged
                        direction: 'fake' | 'real' | 'neutral' | 'limitation'
  4. fuse_verdict     - model probabilities + evidence -> verdict, or
                        UNCERTAIN when they disagree / model is borderline
  5. explain          - GPT explains findings IF key present (never decides);
                        deterministic template otherwise (explainer labelled)

Thresholds below were set from speech literature ranges and verified on
the project's own real/fake segments (see tests/test_evidence.py).
They are deliberately conservative: flags support, never override, the CNN
except to force UNCERTAIN.
"""
import os
import numpy as np
import librosa

SR = 22050
EVIDENCE_MAX_SEC = 20   # cap waveform examined (speed); noted if truncated

# ---------- documented thresholds ----------
F0_MIN, F0_MAX = 50, 500
FLAT_F0_STD = 12.0       # voiced F0 std below this = unnaturally flat prosody
MIN_VOICED_FRAC = 0.15   # below this = barely any speech -> cannot judge
HF_CUT = 4000            # high band edge (Hz)
HF_EXCESS = 0.30         # HF energy share above this = possible vocoder shimmer
FLAT_RMS_STD = 0.015     # loudness variation below this = flat dynamics
NOISE_FLATNESS = 0.25    # spectral flatness above this = noisy recording
CLIP_PEAK = 0.99
UNCERTAIN_LO, UNCERTAIN_HI = 0.45, 0.55   # calibrated fake-prob grey zone


def _decode(path, max_sec=EVIDENCE_MAX_SEC):
    """Decode full(ish) waveform for evidence. Never raises."""
    try:
        y, _ = librosa.load(path, sr=SR, mono=True, duration=max_sec)
        return y
    except Exception:
        pass
    try:
        import sys
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from preprocess import ffmpeg_load
        return ffmpeg_load(path, SR, max_sec)
    except Exception:
        return np.zeros(0, dtype=np.float32)


def validate_audio(path):
    """Return (ok, reason). Rejects files with no usable speech."""
    y = _decode(path, max_sec=10)
    if len(y) < SR:
        return False, 'File is empty or could not be decoded.'
    peak = float(np.abs(y).max())
    if peak < 1e-4:
        return False, 'File is silent (no signal).'
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
    voiced_frac = float((rms > 0.02).mean())
    if voiced_frac < 0.05:
        return False, 'No speech detected (only silence/background).'
    return True, ''


def extract_evidence(path):
    """Compute acoustic evidence scalars. Returns dict (never raises)."""
    ev = {'ok': True, 'reason': '', 'truncated': False}
    y = _decode(path)
    if len(y) < SR:
        return {'ok': False, 'reason': 'Could not decode audio for analysis.'}
    full_len = len(y)
    ev['duration_examined'] = round(full_len / SR, 1)
    # truncated? (compare with container duration via ffprobe, fast)
    try:
        import subprocess
        r = subprocess.run(['ffprobe', '-v', 'error', '-show_entries',
                            'format=duration', '-of', 'csv=p=0', path],
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                           timeout=15, text=True)
        ev['truncated'] = float(r.stdout.strip()) > EVIDENCE_MAX_SEC + 1
    except Exception:
        ev['truncated'] = False

    # ---- dynamics / level ----
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
    ev['peak'] = round(float(np.abs(y).max()), 4)
    ev['rms_mean'] = round(float(rms.mean()), 4)
    ev['rms_std'] = round(float(rms.std()), 4)
    active = rms > 0.02
    ev['voiced_frac'] = round(float(active.mean()), 3)
    ev['silence_frac'] = round(float(1.0 - active.mean()), 3)

    # ---- pitch (yin) on voiced frames ----
    try:
        f0 = librosa.yin(y, fmin=F0_MIN, fmax=F0_MAX, sr=SR,
                         frame_length=2048, hop_length=512)
        # align f0 frames to rms frames (same hop -> same count approx)
        n = min(len(f0), len(rms))
        vf0 = f0[:n][active[:n]]
        vf0 = vf0[(vf0 >= F0_MIN) & (vf0 <= F0_MAX)]
        if len(vf0) > 10:
            ev['f0_mean'] = round(float(vf0.mean()), 1)
            ev['f0_std'] = round(float(vf0.std()), 1)
        else:
            ev['f0_mean'], ev['f0_std'] = None, None
    except Exception:
        ev['f0_mean'], ev['f0_std'] = None, None

    # ---- spectral ----
    try:
        S = np.abs(librosa.stft(y, n_fft=2048)) ** 2
        freqs = librosa.fft_frequencies(sr=SR, n_fft=2048)
        e_total = S.sum() + 1e-12
        ev['hf_ratio'] = round(float(S[freqs >= HF_CUT].sum() / e_total), 3)
        ev['centroid'] = round(float(librosa.feature.spectral_centroid(y=y, sr=SR).mean()), 1)
        ev['rolloff'] = round(float(librosa.feature.spectral_rolloff(y=y, sr=SR).mean()), 1)
        ev['flatness'] = round(float(librosa.feature.spectral_flatness(y=y).mean()), 4)
        ev['zcr'] = round(float(librosa.feature.zero_crossing_rate(y).mean()), 4)
    except Exception:
        ev.update(hf_ratio=None, centroid=None, rolloff=None, flatness=None, zcr=None)

    # ---- MFCC (13 + deltas, summary stats) ----
    try:
        mfcc = librosa.feature.mfcc(y=y, sr=SR, n_mfcc=13)
        d1 = librosa.feature.delta(mfcc)
        ev['mfcc1_mean'] = round(float(mfcc[0].mean()), 2)     # overall tilt
        ev['mfcc_std_mean'] = round(float(mfcc.std(axis=1).mean()), 2)
        ev['delta_mean'] = round(float(np.abs(d1).mean()), 3)  # frame change rate
    except Exception:
        ev.update(mfcc1_mean=None, mfcc_std_mean=None, delta_mean=None)
    return ev


def flag_findings(ev):
    """Turn evidence scalars into tagged findings."""
    F, L = [], []  # findings, limitations
    vf = ev.get('voiced_frac') or 0
    f0std = ev.get('f0_std')
    hf = ev.get('hf_ratio')
    rms_std = ev.get('rms_std')
    flat = ev.get('flatness')
    peak = ev.get('peak') or 0

    if vf < MIN_VOICED_FRAC:
        F.append({'tag': 'LOW SPEECH', 'direction': 'neutral',
                  'text': f'Only {vf:.0%} of frames contain voiced energy - little speech to judge.'})
    if f0std is not None and vf >= MIN_VOICED_FRAC:
        if f0std < FLAT_F0_STD:
            F.append({'tag': 'FLAT PITCH', 'direction': 'fake',
                      'text': f'Pitch varies only {f0std:.1f} Hz (natural speech usually 20-120 Hz) - possible synthetic prosody.'})
        elif 20 <= f0std <= 120:
            F.append({'tag': 'NATURAL PITCH', 'direction': 'real',
                      'text': f'Pitch varies {f0std:.1f} Hz across the clip - consistent with natural prosody.'})
    if hf is not None and hf > HF_EXCESS:
        F.append({'tag': 'HF ENERGY', 'direction': 'fake',
                  'text': f'{hf:.0%} of energy sits above {HF_CUT} Hz - possible vocoder shimmer.'})
    elif hf is not None:
        F.append({'tag': 'CLEAN SPECTRUM', 'direction': 'real',
                  'text': f'High-frequency share normal ({hf:.0%}) - no obvious vocoder shimmer.'})
    if rms_std is not None and vf >= MIN_VOICED_FRAC:
        if rms_std < FLAT_RMS_STD:
            F.append({'tag': 'FLAT DYNAMICS', 'direction': 'fake',
                      'text': 'Loudness barely varies - natural speech breathes more than this.'})
        elif rms_std > 0.02:
            F.append({'tag': 'NATURAL DYNAMICS', 'direction': 'real',
                      'text': 'Loudness dynamics look like natural connected speech.'})
    if flat is not None and flat > NOISE_FLATNESS:
        L.append('High background noise - result less reliable.')
    if peak >= CLIP_PEAK:
        L.append('Clipping detected (distorted peaks) - result less reliable.')
    if ev.get('truncated'):
        L.append('Only first 20 s examined (long file).')
    return F, L


def fuse_verdict(result, ev):
    """Combine CNN probabilities + evidence. Mutates result in place.

    result needs real_probability/fake_probability (0-100).
    Adds: findings, limitations, evidence, explanation, explainer,
          uncertain (bool). May flip label to UNCERTAIN (never to the
    opposite class - the CNN owns the direction).
    """
    p_fake = float(result.get('fake_probability', 50.0)) / 100.0
    findings, limitations = flag_findings(ev)
    fake_flags = sum(1 for f in findings if f['direction'] == 'fake')
    real_flags = sum(1 for f in findings if f['direction'] == 'real')
    vf = ev.get('voiced_frac') or 0

    uncertain = False
    reason = ''
    if UNCERTAIN_LO <= p_fake <= UNCERTAIN_HI:
        uncertain, reason = True, f'model probability {p_fake:.0%} sits in the grey zone'
    elif p_fake < UNCERTAIN_LO and fake_flags >= 2:
        uncertain, reason = True, f'model says human but {fake_flags} acoustic flags disagree'
    elif p_fake > UNCERTAIN_HI and p_fake < 0.65 and fake_flags == 0 and real_flags >= 1:
        uncertain, reason = True, 'weak model signal with no acoustic corroboration'
    elif vf < MIN_VOICED_FRAC:
        uncertain, reason = True, 'too little voiced speech to judge'

    if uncertain:
        result['label'] = 'UNCERTAIN - INCONCLUSIVE'
        result['is_real'] = None
        limitations = [f'Inconclusive: {reason}.'] + limitations
    result['uncertain'] = uncertain
    result['findings'] = findings
    result['limitations'] = limitations
    result['evidence'] = {k: ev.get(k) for k in (
        'f0_mean', 'f0_std', 'voiced_frac', 'silence_frac', 'hf_ratio',
        'centroid', 'rolloff', 'rms_std', 'zcr', 'mfcc1_mean', 'delta_mean',
        'duration_examined')}
    expl = explain_with_gpt(result['label'], findings, p_fake)
    if expl is None:
        expl = explain_rules(result['label'], findings, p_fake, uncertain)
        result['explainer'] = 'rules'
    else:
        result['explainer'] = 'gpt'
    result['explanation'] = expl
    return result


def explain_rules(label, findings, p_fake, uncertain):
    """Deterministic readable summary (used when no GPT key)."""
    n_f = sum(1 for f in findings if f['direction'] == 'fake')
    n_r = sum(1 for f in findings if f['direction'] == 'real')
    bits = []
    for f in findings:
        bits.append(f['text'])
    body = ' '.join(bits) if bits else 'No strong acoustic markers either way.'
    if uncertain:
        return (f'Professor K scores this clip {p_fake:.0%} AI-generated - too close to call. '
                f'Acoustic check: {n_f} synthetic marker(s), {n_r} natural marker(s). {body} '
                f'Retest with a longer, cleaner recording.')
    direction = 'AI-generated' if 'AI' in label else 'human'
    return (f'Professor K scores this clip {p_fake:.0%} AI-generated ({label}, '
            f'{max(p_fake, 1 - p_fake):.0%} confidence). '
            f'Acoustic check: {n_f} synthetic marker(s), {n_r} natural marker(s). {body}')


def explain_with_gpt(label, findings, p_fake):
    """Ask GPT to EXPLAIN given findings. Returns None if unavailable.

    The verdict is already decided - GPT only writes the readable summary.
    Requires OPENAI_API_KEY env var + openai package; any failure -> None.
    """
    key = os.environ.get('OPENAI_API_KEY')
    if not key:
        return None
    try:
        from openai import OpenAI
        client = OpenAI(api_key=key)
        facts = '; '.join(f"[{f['tag']}] {f['text']}" for f in findings) or 'no markers'
        r = client.chat.completions.create(
            model='gpt-4o-mini', temperature=0.2, max_tokens=180,
            messages=[
                {'role': 'system', 'content': (
                    'You explain audio-deepfake evidence to a student. The verdict is ALREADY '
                    'decided and given below - you must NEVER change it, guess, or add new '
                    'claims. Summarise the listed findings in 2-4 plain sentences.')},
                {'role': 'user', 'content': (
                    f'Verdict: {label} (model AI-probability {p_fake:.0%}). '
                    f'Findings: {facts}. Explain briefly.')},
            ])
        return r.choices[0].message.content.strip()
    except Exception:
        return None

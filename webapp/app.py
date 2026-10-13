"""
Voice Impersonation Detection System - Flask Backend

Routes:
    GET  /           -> main page (upload + live record)
    POST /predict    -> analyze audio, return JSON prediction
"""
import os
import sys
import json
import time
import uuid
import traceback

from flask import Flask, request, jsonify, render_template, send_from_directory, Response, stream_with_context
from werkzeug.utils import secure_filename

# allow importing training/ modules (preprocess + model)
# project root: override with VIDS_ROOT env (used by the frozen desktop build,
# where data lives in the PyInstaller bundle temp dir). Dev default unchanged.
ROOT = os.environ.get('VIDS_ROOT') or r"C:\Users\Kishore GK\Desktop\VIDS"
sys.path.insert(0, os.path.join(ROOT, 'training'))

import numpy as np
import torch
from preprocess import audio_to_input, audio_to_chunks, save_mel_image, CLASS_NAMES
from model import VoiceCNN

# ---------------- config ----------------
MODEL_PATH = os.path.join(ROOT, 'model', 'professor_k.pth')
UPLOAD_DIR = os.path.join(ROOT, 'webapp', 'static', 'audio')
ALLOWED = {'.wav', '.mp3', '.m4a', '.ogg', '.flac', '.webm'}
MAX_MB = 20
MIN_SEC = 3
MAX_SEC = 300          # 5 minutes (long audio = split into 3s segments)
SINGLE_MODE_MAX = 15   # <= 15s: single prediction; above: segmented analysis
TEMPERATURE = 2.5      # confidence calibration: softens overconfident 100/0
                       # softmax into honest probabilities (argmax unchanged).
                       # Temporary until multi-generator retrain lands.
SEG_BATCH = 4          # segmented GPU inference batch (progress per batch)

os.makedirs(UPLOAD_DIR, exist_ok=True)

# frozen desktop build serves templates/static from the bundle (one-dir layout
# keeps datas under _internal/); dev keeps defaults
_BUNDLE = None
if getattr(sys, 'frozen', False):
    _mp = getattr(sys, '_MEIPASS', None)
    if _mp:
        _inner = os.path.join(_mp, '_internal')
        _BUNDLE = _inner if os.path.isdir(_inner) else _mp
app = Flask(__name__,
            static_folder=(os.path.join(_BUNDLE, 'webapp', 'static') if _BUNDLE else 'static'),
            template_folder=(os.path.join(_BUNDLE, 'webapp', 'templates') if _BUNDLE else 'templates'))
app.config['MAX_CONTENT_LENGTH'] = MAX_MB * 1024 * 1024

# ---------------- model load ----------------
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
GPU_NAME = torch.cuda.get_device_name(0) if DEVICE == 'cuda' else 'CPU'
model = None
model_epoch = None
model_val_acc = None
MODEL_PARAMS = 0


def load_model():
    global model, model_epoch, model_val_acc, MODEL_PARAMS
    if not os.path.exists(MODEL_PATH):
        print('[warn] model file not found - predictions will be random!')
        return False
    ckpt = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    model = VoiceCNN()
    model.load_state_dict(ckpt['model_state'])
    model.to(DEVICE).eval()
    model_epoch = ckpt.get('epoch')
    model_val_acc = ckpt.get('val_acc')
    MODEL_PARAMS = sum(p.numel() for p in model.parameters())
    # warmup: prove GPU path works before first request
    with torch.no_grad():
        dummy = torch.zeros(1, 1, 128, 128).to(DEVICE)
        if DEVICE == 'cuda':
            torch.cuda.synchronize()
        model(dummy)
        if DEVICE == 'cuda':
            torch.cuda.synchronize()
    print(f'[ok] model "Professor K" loaded ({MODEL_PARAMS:,} params, epoch {model_epoch}, '
          f'val acc {model_val_acc}%) on {DEVICE} [{GPU_NAME}]')


def audio_duration(path):
    """Get duration in seconds (librosa, ffmpeg fallback for .webm)."""
    import librosa
    try:
        return librosa.get_duration(path=path)
    except Exception:
        pass
    try:
        from preprocess import ffmpeg_load, SR
        return len(ffmpeg_load(path, SR)) / SR
    except Exception:
        return None


def gpu_infer(batch):
    """Run Professor K on a numpy batch (N,1,128,128) on GPU.

    Returns (probs ndarray, infer_ms float). Temperature-calibrated
    softmax so displayed probabilities are honest, not 100/0.
    Timing uses cuda.synchronize -> real GPU milliseconds.
    """
    x = torch.from_numpy(batch).to(DEVICE)
    if DEVICE == 'cuda':
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits / TEMPERATURE, dim=1)
    if DEVICE == 'cuda':
        torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) * 1000
    return probs.cpu().numpy(), ms


def sse(data):
    """Format one SSE message."""
    return f"data: {json.dumps(data)}\n\n"


# ---------------- routes ----------------
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():
    """Step 1 of streamed analysis: receive file, return reference name."""
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400
    f = request.files['audio']
    if not f or f.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    ext = os.path.splitext(f.filename)[1].lower()
    if ext not in ALLOWED:
        return jsonify({'error': f'Unsupported format "{ext}". Allowed: {", ".join(sorted(ALLOWED))}'}), 400
    name = f'{uuid.uuid4().hex[:10]}_{secure_filename(f.filename)}'
    path = os.path.join(UPLOAD_DIR, name)
    f.save(path)
    size_mb = os.path.getsize(path) / 1024 / 1024
    return jsonify({'name': name, 'original': f.filename, 'size_mb': round(size_mb, 2)})


@app.route('/predict_stream')
def predict_stream():
    """Step 2 of streamed analysis: SSE progress events + final result.

    Events: {"type":"step","msg":...} ... {"type":"result", <result> }
            {"type":"error","error":...}
    """
    name = request.args.get('name', '')
    if not name or '/' in name or '\\' in name or '..' in name:
        return jsonify({'error': 'Invalid file reference'}), 400
    path = os.path.join(UPLOAD_DIR, name)
    if not os.path.exists(path):
        return jsonify({'error': 'Upload expired - please upload again'}), 404

    def gen():
        t0 = time.time()
        try:
            yield sse({'type': 'step', 'msg': f'file received: {name[11:]}'})
            yield sse({'type': 'step', 'msg': 'decoding audio waveform...'})

            # ---- duration check ----
            t_dec = time.time()
            dur = audio_duration(path)
            dec_ms = round((time.time() - t_dec) * 1000)
            if dur is not None and dur < MIN_SEC:
                yield sse({'type': 'error', 'error': f'Audio too short ({dur:.1f}s). Minimum {MIN_SEC}s required.'})
                return
            if dur is not None and dur > MAX_SEC:
                yield sse({'type': 'error', 'error': f'Audio too long ({dur/60:.1f} min). Maximum {MAX_SEC//60} min allowed.'})
                return
            yield sse({'type': 'step', 'msg': f'waveform: {dur:.1f}s decoded ({dec_ms} ms)'})

            # ---- acoustic validation: reject files with no usable speech ----
            from evidence import validate_audio, extract_evidence, fuse_verdict
            ok, reason = validate_audio(path)
            if not ok:
                yield sse({'type': 'error', 'error': reason})
                return
            yield sse({'type': 'step', 'msg': 'speech validated - extracting acoustic evidence (pitch/spectral/MFCC)...'})
            t_ev = time.time()
            ev = extract_evidence(path)
            ev_ms = round((time.time() - t_ev) * 1000)
            if not ev.get('ok'):
                yield sse({'type': 'error', 'error': ev.get('reason', 'Evidence extraction failed.')})
                return
            f0txt = f"{ev.get('f0_mean')} Hz" if ev.get('f0_mean') else 'unvoiced'
            yield sse({'type': 'step', 'msg': f"evidence ready ({ev_ms} ms): pitch {f0txt}, HF share {ev.get('hf_ratio')}"})

            segmented = dur is not None and dur > SINGLE_MODE_MAX
            result = {}
            total_infer_ms = 0.0

            if not segmented:
                # ===== single mode: full voice-pattern analysis, one window =====
                yield sse({'type': 'step', 'msg': 'building mel-spectrogram 128x128...'})
                t_spec = time.time()
                mel = audio_to_input(path)
                spec_ms = round((time.time() - t_spec) * 1000)
                spec_name = name.rsplit('.', 1)[0] + '_spec.png'
                save_mel_image(mel[0], os.path.join(UPLOAD_DIR, spec_name))
                yield sse({'type': 'step', 'msg': f'spectrogram ready ({spec_ms} ms) - Professor K analyzing on {GPU_NAME}...'})

                if model is None:
                    result = {'label': 'UNKNOWN', 'confidence': 0.0,
                              'real_probability': 50.0, 'fake_probability': 50.0,
                              'warning': 'Model not trained yet - train it first (training/train.py).'}
                else:
                    pr, ms = gpu_infer(mel[np.newaxis, ...] if mel.ndim == 3 else mel)
                    total_infer_ms = ms
                    p_fake, p_real = float(pr[0][0]) * 100, float(pr[0][1]) * 100
                    is_real = p_real >= p_fake
                    yield sse({'type': 'step',
                               'msg': f'GPU inference {ms:.1f} ms - real {p_real:.1f}% vs fake {p_fake:.1f}%'})
                    result = {
                        'label': 'REAL HUMAN VOICE' if is_real else 'AI-GENERATED VOICE',
                        'is_real': bool(is_real),
                        'confidence': round(max(p_real, p_fake), 1),
                        'real_probability': round(p_real, 1),
                        'fake_probability': round(p_fake, 1),
                    }
                result['mode'] = 'single'
                result['spectrogram'] = f'/static/audio/{spec_name}'

            else:
                # ===== segmented mode: split voice into 3s patterns, analyze each =====
                yield sse({'type': 'step', 'msg': f'long audio: splitting into 3s voice patterns...'})
                t_spec = time.time()
                chunks = audio_to_chunks(path, max_sec=MAX_SEC)
                spec_ms = round((time.time() - t_spec) * 1000)
                if not chunks:
                    yield sse({'type': 'error', 'error': 'Could not read audio - invalid or empty file.'})
                    return
                n = len(chunks)
                yield sse({'type': 'step', 'msg': f'{n} blocks ready ({spec_ms} ms) - analyzing patterns on {GPU_NAME}...'})
                spec_name = name.rsplit('.', 1)[0] + '_spec.png'
                save_mel_image(chunks[0][1][0], os.path.join(UPLOAD_DIR, spec_name))

                segments = []
                if model is None:
                    result = {'label': 'UNKNOWN', 'confidence': 0.0,
                              'real_probability': 50.0, 'fake_probability': 50.0,
                              'warning': 'Model not trained yet - train it first (training/train.py).'}
                    for start, _ in chunks:
                        segments.append({'start': round(start, 1), 'label': 'UNKNOWN',
                                         'confidence': 0.0, 'is_real': None})
                else:
                    all_probs = []
                    for i in range(0, n, SEG_BATCH):
                        batch = chunks[i:i + SEG_BATCH]
                        xs = np.stack([m for _, m in batch])
                        pr, ms = gpu_infer(xs)
                        total_infer_ms += ms
                        all_probs.append(pr)
                        for (start, _), p in zip(batch, pr):
                            pf, prl = float(p[0]) * 100, float(p[1]) * 100
                            ir = prl >= pf
                            segments.append({
                                'start': round(start, 1),
                                'label': 'REAL HUMAN' if ir else 'AI-GENERATED',
                                'confidence': round(max(prl, pf), 1),
                                'is_real': bool(ir)})
                        done = min(i + SEG_BATCH, n)
                        tag = segments[-1]['label']
                        yield sse({'type': 'step',
                                   'msg': f'block {done}/{n}: {tag} ({segments[-1]["confidence"]}%) - GPU {ms:.1f} ms'})
                    import numpy as _np
                    all_probs = _np.concatenate(all_probs, axis=0)
                    real_votes = sum(1 for s in segments if s['is_real'])
                    frac_real = real_votes / n
                    overall_real = frac_real >= 0.5
                    conf = max(frac_real, 1 - frac_real) * 100
                    avg_real = float(all_probs[:, 1].mean()) * 100
                    avg_fake = float(all_probs[:, 0].mean()) * 100
                    result = {
                        'label': 'REAL HUMAN VOICE' if overall_real else 'AI-GENERATED VOICE',
                        'is_real': bool(overall_real),
                        'confidence': round(conf, 1),
                        'real_probability': round(avg_real, 1),
                        'fake_probability': round(avg_fake, 1),
                    }
                result['mode'] = 'segmented'
                result['segments'] = segments
                result['n_segments'] = len(segments)
                result['real_votes'] = sum(1 for s in segments if s['is_real'])
                result['spectrogram'] = f'/static/audio/{spec_name}'

            yield sse({'type': 'step', 'msg': 'fusing CNN probabilities + acoustic evidence...'})
            fuse_verdict(result, ev)
            result['evidence_ms'] = ev_ms

            result['duration'] = round(dur, 1) if dur else None
            result['model_epoch'] = model_epoch
            result['model_val_acc'] = model_val_acc
            result['model_name'] = 'Professor K'
            result['device'] = DEVICE
            result['gpu'] = GPU_NAME
            result['infer_ms'] = round(total_infer_ms, 1)
            result['decode_ms'] = dec_ms
            result['spec_ms'] = spec_ms
            result['elapsed'] = round(time.time() - t0, 2)

            yield sse({'type': 'step',
                       'msg': f"verdict: {result.get('label')} ({result.get('confidence')}%)"})
            yield sse({'type': 'result', 'result': result})
        except Exception as e:
            traceback.print_exc()
            yield sse({'type': 'error', 'error': f'Processing failed: {e}'})
        finally:
            try:
                os.remove(path)
            except OSError:
                pass

    return Response(stream_with_context(gen()), mimetype='text/event-stream',
                    headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})


@app.route('/predict', methods=['POST'])
def predict():
    t0 = time.time()

    # ---- 1. input validation ----
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400
    f = request.files['audio']
    if not f or f.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    ext = os.path.splitext(f.filename)[1].lower()
    if ext not in ALLOWED:
        return jsonify({'error': f'Unsupported format "{ext}". Allowed: {", ".join(sorted(ALLOWED))}'}), 400

    # ---- 2. save temp file ----
    name = f'{uuid.uuid4().hex[:10]}_{secure_filename(f.filename)}'
    path = os.path.join(UPLOAD_DIR, name)
    f.save(path)

    try:
        # ---- 3. duration check ----
        dur = audio_duration(path)
        if dur is not None and dur < MIN_SEC:
            return jsonify({'error': f'Audio too short ({dur:.1f}s). Minimum {MIN_SEC}s required.'}), 400
        if dur is not None and dur > MAX_SEC:
            return jsonify({'error': f'Audio too long ({dur/60:.1f} min). Maximum {MAX_SEC//60} min allowed.'}), 400

        # ---- 3b. acoustic validation + evidence (no usable speech -> reject) ----
        from evidence import validate_audio, extract_evidence, fuse_verdict
        ok, reason = validate_audio(path)
        if not ok:
            try:
                os.remove(path)
            except OSError:
                pass
            return jsonify({'error': reason}), 400
        ev = extract_evidence(path)

        # ---- 4. preprocess ----
        segmented = dur is not None and dur > SINGLE_MODE_MAX
        result = {}

        if not segmented:
            # ===== single mode (<=15s): one prediction =====
            mel = audio_to_input(path)
            spec_name = name.rsplit('.', 1)[0] + '_spec.png'
            save_mel_image(mel[0], os.path.join(UPLOAD_DIR, spec_name))

            if model is None:
                result = {
                    'label': 'UNKNOWN', 'confidence': 0.0,
                    'real_probability': 50.0, 'fake_probability': 50.0,
                    'warning': 'Model not trained yet - train it first (training/train.py).',
                }
            else:
                x = torch.from_numpy(mel).unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    probs = torch.softmax(model(x) / TEMPERATURE, dim=1)[0]
                p_fake, p_real = float(probs[0]) * 100, float(probs[1]) * 100
                is_real = p_real >= p_fake
                result = {
                    'label': 'REAL HUMAN VOICE' if is_real else 'AI-GENERATED VOICE',
                    'is_real': bool(is_real),
                    'confidence': round(max(p_real, p_fake), 1),
                    'real_probability': round(p_real, 1),
                    'fake_probability': round(p_fake, 1),
                }
            result['mode'] = 'single'
            result['spectrogram'] = f'/static/audio/{spec_name}'

        else:
            # ===== segmented mode (>15s): split into 3s chunks =====
            chunks = audio_to_chunks(path, max_sec=MAX_SEC)
            if not chunks:
                return jsonify({'error': 'Could not read audio - invalid or empty file.'}), 400

            spec_name = name.rsplit('.', 1)[0] + '_spec.png'
            save_mel_image(chunks[0][1][0], os.path.join(UPLOAD_DIR, spec_name))

            segments = []
            if model is None:
                result = {
                    'label': 'UNKNOWN', 'confidence': 0.0,
                    'real_probability': 50.0, 'fake_probability': 50.0,
                    'warning': 'Model not trained yet - train it first (training/train.py).',
                }
                for start, _ in chunks:
                    segments.append({'start': round(start, 1), 'label': 'UNKNOWN',
                                     'confidence': 0.0, 'is_real': None})
            else:
                # batch predict all chunks at once (fast)
                xs = torch.from_numpy(np.stack([m for _, m in chunks])).to(DEVICE)
                with torch.no_grad():
                    all_probs = torch.softmax(model(xs) / TEMPERATURE, dim=1).cpu().numpy()
                real_votes = 0
                for (start, _), pr in zip(chunks, all_probs):
                    p_fake, p_real = float(pr[0]) * 100, float(pr[1]) * 100
                    is_real = p_real >= p_fake
                    real_votes += int(is_real)
                    segments.append({
                        'start': round(start, 1),
                        'label': 'REAL HUMAN' if is_real else 'AI-GENERATED',
                        'confidence': round(max(p_real, p_fake), 1),
                        'is_real': bool(is_real),
                    })
                # overall = majority vote across segments
                n = len(segments)
                frac_real = real_votes / n
                overall_real = frac_real >= 0.5
                # confidence = dominant class vote share
                conf = max(frac_real, 1 - frac_real) * 100
                # average probability for bars
                avg_real = float(all_probs[:, 1].mean()) * 100
                avg_fake = float(all_probs[:, 0].mean()) * 100
                result = {
                    'label': 'REAL HUMAN VOICE' if overall_real else 'AI-GENERATED VOICE',
                    'is_real': bool(overall_real),
                    'confidence': round(conf, 1),
                    'real_probability': round(avg_real, 1),
                    'fake_probability': round(avg_fake, 1),
                }
            result['mode'] = 'segmented'
            result['segments'] = segments
            result['n_segments'] = len(segments)
            result['real_votes'] = sum(1 for s in segments if s['is_real'])
            result['spectrogram'] = f'/static/audio/{spec_name}'

        # ---- fuse CNN + acoustic evidence (may mark UNCERTAIN) ----
        if ev.get('ok'):
            fuse_verdict(result, ev)

        result['duration'] = round(dur, 1) if dur else None
        result['model_epoch'] = model_epoch
        result['model_val_acc'] = model_val_acc
        result['device'] = DEVICE
        result['elapsed'] = round(time.time() - t0, 2)

        # cleanup audio (keep spectrogram for display)
        try:
            os.remove(path)
        except OSError:
            pass
        return jsonify(result)

    except Exception as e:
        traceback.print_exc()
        try:
            os.remove(path)
        except OSError:
            pass
        return jsonify({'error': f'Processing failed: {e}'}), 500


@app.route('/health')
def health():
    return jsonify({'status': 'ok', 'model_loaded': model is not None,
                    'model_name': 'Professor K', 'params': MODEL_PARAMS,
                    'device': DEVICE, 'gpu': GPU_NAME,
                    'epoch': model_epoch, 'temperature': TEMPERATURE})


# ---------------- main ----------------
if __name__ == '__main__':
    load_model()
    print('Starting web server at http://127.0.0.1:5000')
    app.run(host='127.0.0.1', port=5000, debug=False)

"""Test: upload a 40s audio to /predict and check segmented mode."""
import numpy as np
import soundfile as sf
import json, urllib.request, io

# ---- create 40s test wav ----
sr = 22050
t = np.linspace(0, 40, sr * 40, endpoint=False)
y = 0.4 * np.sin(2 * np.pi * 180 * t) * (0.5 + 0.5 * np.sin(2 * np.pi * 0.5 * t))
y += 0.05 * np.random.randn(len(t))
buf = io.BytesIO()
sf.write(buf, y.astype(np.float32), sr, format='WAV')
raw = buf.getvalue()
print(f'test file: 40s wav ({len(raw)/1e6:.1f} MB)')

# ---- multipart POST ----
boundary = '----test123'
body = (
    b'--' + boundary.encode() + b'\r\n'
    b'Content-Disposition: form-data; name="audio"; filename="test40.wav"\r\n'
    b'Content-Type: audio/wav\r\n\r\n'
    + raw + b'\r\n'
    + b'--' + boundary.encode() + b'--\r\n'
)
req = urllib.request.Request(
    'http://127.0.0.1:5000/predict', data=body,
    headers={'Content-Type': 'multipart/form-data; boundary=' + boundary})
r = json.load(urllib.request.urlopen(req, timeout=120))

print('mode       :', r.get('mode'))
print('n_segments :', r.get('n_segments'))
print('label      :', r.get('label'), '| confidence:', r.get('confidence'))
print('duration   :', r.get('duration'), 's | elapsed:', r.get('elapsed'), 's')
segs = r.get('segments', [])
print('segments[0:3]:', segs[:3])
print('real_votes :', r.get('real_votes'))
print('warning    :', r.get('warning'))
print('error      :', r.get('error'))
assert r.get('mode') == 'segmented', 'FAIL: not segmented mode'
assert r.get('n_segments', 0) >= 12, 'FAIL: expected ~13 chunks for 40s'
print('\nLONG-AUDIO TEST PASSED')

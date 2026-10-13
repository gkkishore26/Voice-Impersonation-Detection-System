"""E2E test on BALANCED dataset files (both classes = full 3s speech)."""
import os, glob, json, urllib.request

BAL = r"C:\Users\Kishore GK\Desktop\VIDS\dataset\balanced"
BOUNDARY = '----vidstest2'

def upload(path):
    with open(path, 'rb') as f:
        raw = f.read()
    fname = os.path.basename(path)
    body = (
        b'--' + BOUNDARY.encode() + b'\r\n'
        b'Content-Disposition: form-data; name="audio"; filename="' + fname.encode() + b'"\r\n'
        b'Content-Type: audio/wav\r\n\r\n' + raw + b'\r\n'
        + b'--' + BOUNDARY.encode() + b'--\r\n'
    )
    req = urllib.request.Request(
        'http://127.0.0.1:5000/predict', data=body,
        headers={'Content-Type': 'multipart/form-data; boundary=' + BOUNDARY})
    return json.load(urllib.request.urlopen(req, timeout=120))

def report(name, path, expected):
    r = upload(path)
    label = r.get('label', '?')
    conf = r.get('confidence', 0)
    ok = (expected == 'real' and 'REAL' in label) or (expected == 'fake' and 'AI' in label)
    mark = 'CORRECT' if ok else 'WRONG '
    print(f'[{mark}] {name:22s} expected {expected:4s} -> {label} ({conf}%)  {r.get("elapsed")}s')
    return ok

import time
time.sleep(2)  # wait for flask

print('=== Professor K - Honest Model End-to-End Test (balanced data) ===\n')
results = []

# test on held-out balanced files
for p in sorted(glob.glob(os.path.join(BAL, 'real', '*.wav')))[:5]:
    results.append(report('REAL segment', p, 'real'))
for p in sorted(glob.glob(os.path.join(BAL, 'fake', '*.wav')))[:5]:
    results.append(report('FAKE segment', p, 'fake'))

n_ok = sum(results)
print(f'\nRESULT: {n_ok}/{len(results)} correct ({100*n_ok/len(results):.0f}%)')

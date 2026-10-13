"""End-to-end test: upload real & fake clips to web app, check Professor K verdicts."""
import os, glob, json, urllib.request

ROOT = r"C:\Users\Kishore GK\Desktop\VIDS\dataset\raw"
BOUNDARY = '----vidstest'

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
    real_p = r.get('real_probability', 0)
    fake_p = r.get('fake_probability', 0)
    ok = (expected == 'real' and 'REAL' in label) or (expected == 'fake' and 'AI' in label)
    mark = 'CORRECT' if ok else 'WRONG'
    print(f'[{mark}] {name}: expected {expected.upper():4s} -> {label} ({conf}%)  '
          f'[real {real_p}% | fake {fake_p}%]  {r.get("elapsed")}s')
    return ok

print('=== Professor K End-to-End Prediction Test ===\n')
results = []

reals = sorted(glob.glob(os.path.join(ROOT, 'real', '*.wav')))[:4]
fakes = sorted(glob.glob(os.path.join(ROOT, 'fake', '*.wav')))[:4]

for p in reals:
    results.append(report('REAL file', p, 'real'))
for p in fakes:
    results.append(report('FAKE file', p, 'fake'))

n_ok = sum(results)
print(f'\nRESULT: {n_ok}/{len(results)} correct '
      f'({100*n_ok/len(results):.0f}% accuracy on unseen files)')

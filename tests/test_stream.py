"""Test streamed analysis: upload -> SSE steps -> final result."""
import glob, json, os, urllib.request

BAL = r"C:\Users\Kishore GK\Desktop\VIDS\dataset\balanced"
BOUNDARY = '----vidssse'

def post_upload(path):
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
        'http://127.0.0.1:5000/upload', data=body,
        headers={'Content-Type': 'multipart/form-data; boundary=' + BOUNDARY})
    return json.load(urllib.request.urlopen(req, timeout=60))

def stream_predict(name):
    req = urllib.request.Request(
        'http://127.0.0.1:5000/predict_stream?name=' + name)
    final = None
    with urllib.request.urlopen(req, timeout=180) as r:
        buf = b''
        while True:
            chunk = r.read(1024)
            if not chunk:
                break
            buf += chunk
            while b'\n\n' in buf:
                msg, buf = buf.split(b'\n\n', 1)
                for line in msg.decode().splitlines():
                    if line.startswith('data: '):
                        m = json.loads(line[6:])
                        if m['type'] == 'step':
                            print(f"  | {m['msg']}")
                        elif m['type'] == 'result':
                            final = m['result']
                        elif m['type'] == 'error':
                            print(f"  !! ERROR: {m['error']}")
    return final

import time
time.sleep(6)  # wait for flask + model load

print('=== SSE Stream Test: REAL file ===')
real = sorted(glob.glob(os.path.join(BAL, 'real', '*.wav')))[0]
u = post_upload(real)
print(f"upload ok: {u['original']} ({u['size_mb']} MB)")
r = stream_predict(u['name'])
print(f"=> {r['label']} real={r['real_probability']}% fake={r['fake_probability']}%")
print(f"=> device={r['device']} gpu={r['gpu']} infer={r['infer_ms']}ms decode={r['decode_ms']}ms spec={r['spec_ms']}ms total={r['elapsed']}s")

print('\n=== SSE Stream Test: FAKE file ===')
fake = sorted(glob.glob(os.path.join(BAL, 'fake', '*.wav')))[0]
u = post_upload(fake)
print(f"upload ok: {u['original']} ({u['size_mb']} MB)")
r = stream_predict(u['name'])
print(f"=> {r['label']} real={r['real_probability']}% fake={r['fake_probability']}%")
print(f"=> device={r['device']} gpu={r['gpu']} infer={r['infer_ms']}ms total={r['elapsed']}s")

print('\n=== /health ===')
h = json.load(urllib.request.urlopen('http://127.0.0.1:5000/health', timeout=10))
print(h)

"""Verify installed VIDS.exe: health + one real-file prediction via its backend."""
import json, time, urllib.request

BASE = 'http://127.0.0.1:5000'
ok = False
for _ in range(90):
    try:
        h = json.load(urllib.request.urlopen(BASE + '/health', timeout=5))
        ok = True
        break
    except Exception:
        time.sleep(2)
assert ok, 'installed backend never came up'
print('INSTALLED HEALTH:', {k: h.get(k) for k in ('model_loaded', 'epoch', 'device', 'model_name')})

B = '----insttest'
path = r"C:\Users\Kishore GK\Desktop\VIDS\dataset\balanced\real\blr_00000.wav"
with open(path, 'rb') as f:
    raw = f.read()
body = (b'--' + B.encode() + b'\r\n'
        b'Content-Disposition: form-data; name="audio"; filename="t.wav"\r\n'
        b'Content-Type: audio/wav\r\n\r\n' + raw + b'\r\n'
        + b'--' + B.encode() + b'--\r\n')
req = urllib.request.Request(BASE + '/upload', data=body,
    headers={'Content-Type': 'multipart/form-data; boundary=' + B})
u = json.load(urllib.request.urlopen(req, timeout=120))
req2 = urllib.request.Request(BASE + '/predict_stream?name=' + u['name'])
final = None
with urllib.request.urlopen(req2, timeout=600) as r:
    buf = b''
    while True:
        ch = r.read(1024)
        if not ch:
            break
        buf += ch
        while b'\n\n' in buf:
            msg, buf = buf.split(b'\n\n', 1)
            for line in msg.decode().splitlines():
                if line.startswith('data: '):
                    m = json.loads(line[6:])
                    if m['type'] == 'result':
                        final = m['result']
print('INSTALLED PREDICT:', final['label'], '| real', final['real_probability'],
      '% | findings:', len(final.get('findings', [])))
assert 'REAL' in final['label'], 'wrong verdict!'
print('INSTALL TEST: PASS')

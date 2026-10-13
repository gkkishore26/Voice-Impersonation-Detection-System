"""Fake-file check against INSTALLED app backend."""
import json, urllib.request

BASE = 'http://127.0.0.1:5000'
B = '----instfake'
path = r"C:\Users\Kishore GK\Downloads\femail ai.mp3"
with open(path, 'rb') as f:
    raw = f.read()
body = (b'--' + B.encode() + b'\r\n'
        b'Content-Disposition: form-data; name="audio"; filename="t.mp3"\r\n'
        b'Content-Type: audio/mpeg\r\n\r\n' + raw + b'\r\n'
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
print('INSTALLED FAKE:', final['label'], '| fake', final['fake_probability'],
      '% | findings:', len(final.get('findings', [])))

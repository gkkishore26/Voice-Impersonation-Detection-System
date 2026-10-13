"""Field-test the deployed model on the 2 user ElevenLabs files (full-length)."""
import json, urllib.request

B = '----ftcheck'
FILES = [
    (r"C:\Users\Kishore GK\Downloads\motivation is a myth.mp3", 'motivation-myth'),
    (r"C:\Users\Kishore GK\Downloads\male ai.mp3", 'male-ai-NEW'),
    (r"C:\Users\Kishore GK\Downloads\femail ai.mp3", 'female-ai-NEW'),
]

for path, tag in FILES:
    with open(path, 'rb') as f:
        raw = f.read()
    head = b'--' + B.encode() + b'\r\n'
    head += b'Content-Disposition: form-data; name="audio"; filename="t.mp3"\r\n'
    head += b'Content-Type: audio/mpeg\r\n\r\n'
    body = head + raw + b'\r\n--' + B.encode() + b'--\r\n'
    req = urllib.request.Request(
        'http://127.0.0.1:5000/upload', data=body,
        headers={'Content-Type': 'multipart/form-data; boundary=' + B})
    u = json.load(urllib.request.urlopen(req, timeout=120))
    req2 = urllib.request.Request(
        'http://127.0.0.1:5000/predict_stream?name=' + u['name'])
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
    print(tag, '=>', final['label'], '| fake', final['fake_probability'],
          '% | uncertain:', final.get('uncertain'),
          '| findings:', len(final.get('findings', [])))

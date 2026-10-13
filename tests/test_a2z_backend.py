"""A2Z backend audit: health, validation errors, single/segmented/webm predictions."""
import json, urllib.request, urllib.error

BASE = 'http://127.0.0.1:5000'
B = '----a2z'
results = []


def check(name, ok, detail=''):
    results.append((name, ok, detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")


def post_file(url, path, fname, ctype):
    with open(path, 'rb') as f:
        raw = f.read()
    body = (b'--' + B.encode() + b'\r\n'
            b'Content-Disposition: form-data; name="audio"; filename="' + fname.encode() + b'"\r\n'
            b'Content-Type: ' + ctype.encode() + b'\r\n\r\n' + raw + b'\r\n'
            + b'--' + B.encode() + b'--\r\n')
    req = urllib.request.Request(url, data=body,
        headers={'Content-Type': 'multipart/form-data; boundary=' + B})
    try:
        r = urllib.request.urlopen(req, timeout=120)
        return r.status, json.load(r)
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode())


def stream(name):
    req = urllib.request.Request(BASE + '/predict_stream?name=' + name)
    final, steps, err = None, 0, None
    with urllib.request.urlopen(req, timeout=600) as r:
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
                        if m['type'] == 'step':
                            steps += 1
                        elif m['type'] == 'result':
                            final = m['result']
                        elif m['type'] == 'error':
                            err = m['error']
    return final, steps, err


# 1. health
h = json.load(urllib.request.urlopen(BASE + '/health', timeout=15))
check('health: model loaded', h.get('model_loaded') is True, str({k: h.get(k) for k in ('epoch', 'device', 'model_name')}))

# 2. validation errors
s, j = post_file(BASE + '/upload', r"C:\Users\Kishore GK\Desktop\VIDS\dataset\balanced\real\blr_00000.wav", 'x.txt', 'text/plain')
check('reject bad extension', s == 400, f'status={s}')
s, j = post_file(BASE + '/predict', r"C:\Users\Kishore GK\Desktop\VIDS\dataset\balanced\real\blr_00000.wav", 'x.wav', 'audio/wav')
check('legacy /predict alive', s == 200 and 'label' in j, f"label={j.get('label')}")

# 3. single predictions (real + fake)
for cls, expect in (('real', 'REAL'), ('fake', 'AI')):
    p = rf"C:\Users\Kishore GK\Desktop\VIDS\dataset\balanced\{cls}\blr_00000.wav"
    s, u = post_file(BASE + '/upload', p, 't.wav', 'audio/wav')
    final, steps, err = stream(u['name'])
    ok = final and expect in final['label'] and steps >= 5 and final.get('evidence') and final.get('findings') is not None
    check(f'single {cls} verdict+evidence+steps', bool(ok),
          f"{final['label']} {final['confidence']}% steps={steps} findings={len(final.get('findings', []))}" if final else f'err={err}')

# 4. segmented (long file > 15s)
p = r"C:\Users\Kishore GK\Downloads\motivation is a myth.mp3"
s, u = post_file(BASE + '/upload', p, 'long.mp3', 'audio/mpeg')
final, steps, err = stream(u['name'])
ok = final and final.get('mode') == 'segmented' and final.get('n_segments', 0) > 5
check('segmented long file', bool(ok),
      f"mode={final.get('mode')} segs={final.get('n_segments')} verdict={final.get('label')}" if final else f'err={err}')

# 5. webm path
import subprocess
webm = r"C:\Users\Kishore GK\AppData\Local\Temp\a2z_rec.webm"
subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i',
                r"C:\Users\Kishore GK\Desktop\VIDS\dataset\balanced\real\blr_00001.wav",
                '-c:a', 'libopus', webm], check=True)
s, u = post_file(BASE + '/upload', webm, 'live_recording.webm', 'audio/webm')
final, steps, err = stream(u['name'])
check('webm recording path', bool(final and 'REAL' in final['label']), final['label'] if final else f'err={err}')

# 6. short file rejected
import wave
shortp = r"C:\Users\Kishore GK\AppData\Local\Temp\a2z_short.wav"
with wave.open(shortp, 'wb') as w:
    w.setnchannels(1); w.setsampwidth(2); w.setframerate(22050)
    w.writeframes(b'\x00\x00' * 22050)
s, u = post_file(BASE + '/upload', shortp, 's.wav', 'audio/wav')
final, steps, err = stream(u['name'])
check('short file rejected', err is not None and 'short' in err.lower(), f'err={err}')

n_ok = sum(1 for _, ok, _ in results if ok)
print(f'\nBACKEND: {n_ok}/{len(results)} passed')

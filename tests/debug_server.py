"""Run the BUNDLED app server (threaded, like the exe) + full prediction."""
import os
import sys
import threading
import time
import urllib.request
import json

B = r"C:\Users\Kishore GK\Desktop\VIDS\Software\dist\VIDS\_internal"
for sub in ('', 'webapp', 'training'):
    p = os.path.join(B, sub) if sub else B
    sys.path.insert(0, p)
os.environ['VIDS_ROOT'] = B
os.environ['PATH'] = B + os.pathsep + os.environ.get('PATH', '')

import app
print('load:', app.load_model(), flush=True)
threading.Thread(target=lambda: app.app.run(
    host='127.0.0.1', port=5000, debug=False,
    threaded=True, use_reloader=False), daemon=True).start()

ok = False
for _ in range(60):
    try:
        h = json.load(urllib.request.urlopen('http://127.0.0.1:5000/health', timeout=5))
        ok = True
        break
    except Exception:
        time.sleep(2)
print('HEALTH:', h if ok else 'NEVER UP', flush=True)
assert ok

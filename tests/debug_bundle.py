"""Replicate frozen runtime against dist bundle (debug installed app)."""
import os
import sys

B = r"C:\Users\Kishore GK\Desktop\VIDS\Software\dist\VIDS\_internal"
for sub in ('', 'webapp', 'training'):
    p = os.path.join(B, sub) if sub else B
    if p not in sys.path:
        sys.path.insert(0, p)
os.environ['VIDS_ROOT'] = B
os.environ['PATH'] = B + os.pathsep + os.environ.get('PATH', '')

import app
print('import app OK; ROOT =', app.ROOT)
print('model file:', os.path.exists(app.MODEL_PATH), app.MODEL_PATH)
print('load_model:', app.load_model())
print('device:', app.DEVICE, '| model:', app.model)

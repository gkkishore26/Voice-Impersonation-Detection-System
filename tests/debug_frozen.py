"""Replicate frozen bundle runtime step-by-step (find the hang)."""
import os
import sys
import time

B = r"C:\Users\Kishore GK\Desktop\VIDS\Software\dist\VIDS\_internal"
for sub in ('', 'webapp', 'training'):
    p = os.path.join(B, sub) if sub else B
    sys.path.insert(0, p)
os.environ['VIDS_ROOT'] = B
os.environ['PATH'] = B + os.pathsep + os.environ.get('PATH', '')

t0 = time.time()
print('t+0 importing app...', flush=True)
import app
print(f't+{time.time()-t0:.1f}s import OK', flush=True)
print('t+0 loading model...', flush=True)
t0 = time.time()
app.load_model()
print(f't+{time.time()-t0:.1f}s load_model done, device={app.DEVICE}', flush=True)
print('t+0 test inference...', flush=True)
t0 = time.time()
import numpy as np
import torch
x = torch.zeros(1, 1, 128, 128)
with torch.no_grad():
    out = app.model(x.to(app.DEVICE))
print(f't+{time.time()-t0:.1f}s inference OK:', out.shape, flush=True)

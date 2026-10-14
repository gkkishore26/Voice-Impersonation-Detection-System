import os
import json
import subprocess
import urllib.request
import urllib.error

T = os.environ.get('GITHUB_PAT')
if not T:
    T = subprocess.run(
        ['powershell', '-noprofile', '-command',
         "[System.Environment]::GetEnvironmentVariable('GITHUB_PAT','User')"],
        capture_output=True, text=True).stdout.strip()

url = 'https://api.github.com/repos/gkkishore26/Voice-Impersonation-Detection-System/releases'
req = urllib.request.Request(url, headers={
    'Authorization': 'Bearer ' + T,
    'Accept': 'application/vnd.github+json'})
try:
    r = urllib.request.urlopen(req, timeout=20)
    print('GET releases:', r.status, 'OK (read works)')
except urllib.error.HTTPError as e:
    print('GET releases:', e.code, e.reason)

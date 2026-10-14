import json
import subprocess
import urllib.request

T = subprocess.run(
    ['powershell', '-noprofile', '-command',
     "[System.Environment]::GetEnvironmentVariable('GITHUB_PAT','User')"],
    capture_output=True, text=True).stdout.strip()
req = urllib.request.Request(
    'https://api.github.com/repos/gkkishore26/Voice-Impersonation-Detection-System/releases/tags/v1.0',
    headers={'Authorization': 'Bearer ' + T,
             'Accept': 'application/vnd.github+json'})
rel = json.load(urllib.request.urlopen(req, timeout=30))
b = rel.get('body', '')
print('personal hits in release body:',
      [w for w in ['KISHORE', 'Kishore', '24ADU', 'Karpagam', 'gmail', 'Gnanakumar']
       if w.lower() in b.lower()] or 'NONE')
r = json.load(urllib.request.urlopen(urllib.request.Request(
    'https://api.github.com/repos/gkkishore26/Voice-Impersonation-Detection-System',
    headers={'Authorization': 'Bearer ' + T,
             'Accept': 'application/vnd.github+json'}), timeout=30))
print('repo description:', r.get('description'))

import json
import subprocess
import urllib.request

T = subprocess.run(
    ['powershell', '-noprofile', '-command',
     "[System.Environment]::GetEnvironmentVariable('GITHUB_PAT','User')"],
    capture_output=True, text=True).stdout.strip()
req = urllib.request.Request(
    'https://api.github.com/repos/gkkishore26/Voice-Impersonation-Detection-System/contents/',
    headers={'Authorization': 'Bearer ' + T,
             'Accept': 'application/vnd.github+json'})
items = json.load(urllib.request.urlopen(req, timeout=20))
print(sorted(it['name'] for it in items))

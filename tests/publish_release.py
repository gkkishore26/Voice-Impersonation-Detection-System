"""Create GitHub Release v1.0 and upload Setup.exe as an asset.

Usage: python publish_release.py  (reads GITHUB_PAT from user env registry)
"""
import json
import os
import urllib.request

OWNER, REPO = 'gkkishore26', 'Voice-Impersonation-Detection-System'
TAG = 'v1.0'
SETUP = r"C:\Users\Kishore GK\Desktop\VIDS\Software\Setup.exe"


def api(url, data=None, headers=None, method=None):
    h = {'Authorization': 'Bearer ' + TOKEN, 'Accept': 'application/vnd.github+json'}
    if headers:
        h.update(headers)
    req = urllib.request.Request(url, data=data, headers=h, method=method)
    return json.load(urllib.request.urlopen(req, timeout=120))


def main():
    # 1. create (or reuse) release
    try:
        rel = api(f'https://api.github.com/repos/{OWNER}/{REPO}/releases',
                  data=json.dumps({
                      'tag_name': TAG,
                      'name': 'VIDS v1.0 - Voice Impersonation Detection System',
                      'body': ('Windows installer (no admin needed).\n\n'
                               'Professor K CNN (4.58M params): val 94.6%, ROC-AUC 0.99, '
                               '16/16 held-out file test.\n'
                               'Upload/record a voice clip - real human or AI-generated verdict '
                               'with acoustic evidence, spectrogram and printable report.'),
                      'draft': False,
                      'prerelease': False,
                  }).encode())
        print('release created:', rel['id'])
    except urllib.error.HTTPError as e:
        if e.code == 422:  # release/tag exists - reuse it
            rel = api(f'https://api.github.com/repos/{OWNER}/{REPO}/releases/tags/{TAG}')
            print('release exists, reusing:', rel['id'])
        else:
            raise
    # 2. delete stale Setup.exe asset if present (idempotent re-runs)
    assets = api(rel['url'] + '/assets')
    for a in assets:
        if a['name'] == 'Setup.exe':
            req = urllib.request.Request(
                a['url'], method='DELETE',
                headers={'Authorization': 'Bearer ' + TOKEN,
                         'Accept': 'application/vnd.github+json'})
            urllib.request.urlopen(req, timeout=120).read()
            print('deleted stale asset')
    # 3. upload via curl (streams 400MB+ correctly; urllib mangles it)
    import subprocess
    size = os.path.getsize(SETUP)
    print(f'uploading Setup.exe ({size / 1e6:.0f} MB)...', flush=True)
    up_url = rel['upload_url'].split('{')[0] + '?name=Setup.exe'
    out = subprocess.run(
        ['curl', '-sS', '-X', 'POST',
         '-H', 'Authorization: Bearer ' + TOKEN,
         '-H', 'Content-Type: application/octet-stream',
         '--data-binary', '@' + SETUP, up_url],
        capture_output=True, text=True, timeout=3600)
    if out.returncode != 0:
        print('curl failed:', out.stderr[-2000:])
        raise SystemExit(1)
    info = json.loads(out.stdout)
    print('UPLOADED:', info['name'], info['size'], 'bytes')
    print('download:', info['browser_download_url'])


if __name__ == '__main__':
    import urllib.error  # noqa: E402
    TOKEN = os.environ.get('GITHUB_PAT') or __import__('subprocess').run(
        ['powershell', '-noprofile', '-command',
         "[System.Environment]::GetEnvironmentVariable('GITHUB_PAT','User')"],
        capture_output=True, text=True).stdout.strip()
    assert TOKEN, 'GITHUB_PAT not found'
    main()

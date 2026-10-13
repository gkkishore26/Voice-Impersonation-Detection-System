# Download real vs fake voice dataset from HuggingFace
import urllib.request, urllib.parse, json, os, time, sys

BASE = r"C:\Users\Kishore GK\Desktop\VIDS\dataset\raw"
REPO = "3004lakshu/Deepfake-Audio"
API = f"https://huggingface.co/api/datasets/{REPO}/tree/main?recursive=true"

def list_files():
    req = urllib.request.Request(API, headers={'User-Agent': 'Mozilla/5.0'})
    data = json.load(urllib.request.urlopen(req, timeout=30))
    files = []
    for f in data:
        if f['type'] != 'file':
            continue
        p = f['path']
        if not p.startswith('raw_audio/') or p.endswith('.DS_Store'):
            continue
        if '/fake/' in p:
            files.append((p, os.path.join(BASE, 'fake', os.path.basename(p))))
        elif '/real/' in p:
            files.append((p, os.path.join(BASE, 'real', os.path.basename(p))))
    return files

def download(url, dest, retries=3):
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=60) as r, open(dest, 'wb') as out:
                out.write(r.read())
            return True
        except Exception as e:
            if i == retries - 1:
                print('FAIL', os.path.basename(dest), str(e)[:80])
                return False
            time.sleep(2)

def main():
    files = list_files()
    print(f'Total files to download: {len(files)}')
    ok = fail = 0
    for i, (repo_path, dest) in enumerate(files, 1):
        if os.path.exists(dest) and os.path.getsize(dest) > 0:
            ok += 1
            continue
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        url = f"https://huggingface.co/datasets/{REPO}/resolve/main/" + urllib.parse.quote(repo_path)
        if download(url, dest):
            ok += 1
        else:
            fail += 1
        if i % 50 == 0:
            print(f'  progress: {i}/{len(files)} (ok={ok}, fail={fail})')
    print(f'DONE: ok={ok}, fail={fail}')
    for cls in ('fake', 'real'):
        d = os.path.join(BASE, cls)
        n = len([f for f in os.listdir(d) if not f.endswith('.DS_Store')]) if os.path.isdir(d) else 0
        size = sum(os.path.getsize(os.path.join(d, f)) for f in os.listdir(d)) if os.path.isdir(d) else 0
        print(f'  {cls}: {n} files, {size/1e6:.1f} MB')

if __name__ == '__main__':
    main()

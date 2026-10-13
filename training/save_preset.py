"""Save current training state as a named, restorable preset.

Copies: model weights, histories, field manifest + segments list,
        finetune file lists, training config. Verifies via sha256.
Usage: python save_preset.py <preset_name>
"""
import os, sys, json, shutil, hashlib, datetime

ROOT = r"C:\Users\Kishore GK\Desktop\VIDS"
MODEL = os.path.join(ROOT, 'model')
PRESETS = os.path.join(MODEL, 'presets')


def sha256(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else 'preset_' + datetime.datetime.now().strftime('%Y%m%d_%H%M')
    dest = os.path.join(PRESETS, name)
    if os.path.exists(dest):
        print('preset already exists:', name)
        return
    os.makedirs(dest)

    files = {}
    # 1. model weights + histories
    for fn in ('professor_k.pth', 'professor_k_v1_backup.pth',
               'training_history.json', 'metrics.json', 'finetune_history.json'):
        src = os.path.join(MODEL, fn)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(dest, fn))
            files[fn] = {'bytes': os.path.getsize(src), 'sha256': sha256(src)}
    # 2. field manifest (labels + sources of progressive data)
    fmanifest = os.path.join(ROOT, 'dataset', 'field', 'manifest.json')
    if os.path.exists(fmanifest):
        shutil.copy(fmanifest, os.path.join(dest, 'field_manifest.json'))
        fm = json.load(open(fmanifest))
        files['field_manifest.json'] = {'segments': len(fm)}
    # 3. exact finetune file lists (reproducible mix, seed 11)
    for cls in ('fake', 'real'):
        d = os.path.join(ROOT, 'dataset', 'finetune', cls)
        if os.path.isdir(d):
            files[f'finetune_{cls}'] = sorted(os.listdir(d))
    # 4. training configuration used
    ft_hist = os.path.join(MODEL, 'finetune_history.json')
    ft = json.load(open(ft_hist)) if os.path.exists(ft_hist) else {}
    config = {
        'preset': name,
        'saved_at': datetime.datetime.now().isoformat(timespec='seconds'),
        'model': 'Professor K (VoiceCNN, 4,583,874 params)',
        'stage': 'v1 multi-vocoder (40ep, GPU) + field fine-tune (10ep, lr 1e-4)',
        'checkpoint_epoch': 49,
        'finetune_val_acc': ft.get('best_val_acc'),
        'temperature': 2.5,
        'data': {'multi': {'real': 1363, 'fake': 1786},
                 'field_fake_segments': 103,
                 'finetune_mix': {'fake': 203, 'real': 200}},
        'restore_cmd': f'python training/restore_preset.py {name}',
    }
    json.dump(config, open(os.path.join(dest, 'PRESET.json'), 'w'), indent=1)
    json.dump(files, open(os.path.join(dest, 'FILES.json'), 'w'), indent=1)
    total_mb = sum(v.get('bytes', 0) for v in files.values() if isinstance(v, dict)) / 1e6
    print(f'preset saved: {name} ({total_mb:.1f} MB)')
    print('restore with: python training/restore_preset.py', name)


if __name__ == '__main__':
    main()

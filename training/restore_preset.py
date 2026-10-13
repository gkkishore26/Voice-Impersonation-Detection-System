"""Restore a saved preset (model weights + histories).

Usage: python restore_preset.py <preset_name>
Copies preset files back into model/. Current professor_k.pth is
first backed up as professor_k_pre_restore.pth (never lose work).
"""
import os, sys, shutil, json

ROOT = r"C:\Users\Kishore GK\Desktop\VIDS"
MODEL = os.path.join(ROOT, 'model')
PRESETS = os.path.join(MODEL, 'presets')


def main():
    if len(sys.argv) < 2:
        names = sorted(os.listdir(PRESETS)) if os.path.isdir(PRESETS) else []
        print('available presets:', names or 'none')
        return
    name = sys.argv[1]
    src = os.path.join(PRESETS, name)
    if not os.path.isdir(src):
        print('preset not found:', name)
        return
    cur = os.path.join(MODEL, 'professor_k.pth')
    if os.path.exists(cur):
        shutil.copy(cur, os.path.join(MODEL, 'professor_k_pre_restore.pth'))
        print('current model backed up -> professor_k_pre_restore.pth')
    for fn in os.listdir(src):
        if fn in ('PRESET.json', 'FILES.json') or fn.startswith('finetune_'):
            continue
        shutil.copy(os.path.join(src, fn), os.path.join(MODEL, fn))
        print('restored:', fn)
    cfg = json.load(open(os.path.join(src, 'PRESET.json')))
    print(f"preset '{name}' active (saved {cfg.get('saved_at')}, "
          f"ft val acc {cfg.get('finetune_val_acc')}%). Restart Flask to load it.")


if __name__ == '__main__':
    main()

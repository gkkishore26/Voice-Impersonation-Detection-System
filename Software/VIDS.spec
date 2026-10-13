# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['src/desktop_app.py'],
    pathex=['C:/Users/Kishore GK/Desktop/VIDS/webapp', 'C:/Users/Kishore GK/Desktop/VIDS/training'],
    binaries=[],
    datas=[('C:/Users/Kishore GK/Desktop/VIDS/webapp/templates', 'webapp/templates'), ('C:/Users/Kishore GK/Desktop/VIDS/webapp/static', 'webapp/static'), ('C:/Users/Kishore GK/Desktop/VIDS/webapp/app.py', 'webapp'), ('C:/Users/Kishore GK/Desktop/VIDS/training', 'training'), ('C:/Users/Kishore GK/Desktop/VIDS/model/professor_k.pth', 'model'), ('C:/Users/Kishore GK/AppData/Local/Microsoft/WinGet/Packages/Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe/ffmpeg-8.1-full_build/bin/ffmpeg.exe', '.'), ('C:/Users/Kishore GK/AppData/Local/Microsoft/WinGet/Packages/Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe/ffmpeg-8.1-full_build/bin/ffprobe.exe', '.')],
    hiddenimports=['soundfile', 'audioread', 'matplotlib', 'PIL'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['PySide6', 'PyQt5', 'PyQt6', 'tkinter'],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='VIDS',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=['C:/Users/Kishore GK/Desktop/VIDS/Software/logo.ico'],
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='VIDS',
)

@echo off
REM VIDS desktop build (PyInstaller onedir). Run from VIDS project root.
cd /d "%~dp0.."
set FFMPEG_BIN=C:\Users\Kishore GK\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.1-full_build\bin
set "PYBIN=%CD%\Software\exe_env\Scripts\python.exe"
"%PYBIN%" -m PyInstaller --noconfirm --clean --onedir --windowed --name VIDS ^
  --icon "%CD%\Software\logo.ico" ^
  --distpath Software\dist --workpath Software\build --specpath Software ^
  --paths "%CD%\webapp" --paths "%CD%\training" ^
  --exclude-module PySide6 --exclude-module PyQt5 --exclude-module PyQt6 ^
  --exclude-module tkinter ^
  --add-data "%CD%\webapp\templates;webapp\templates" ^
  --add-data "%CD%\webapp\static;webapp\static" ^
  --add-data "%CD%\webapp\app.py;webapp" ^
  --add-data "%CD%\training;training" ^
  --add-data "%CD%\model\professor_k.pth;model" ^
  --add-data "%FFMPEG_BIN%\ffmpeg.exe;." ^
  --add-data "%FFMPEG_BIN%\ffprobe.exe;." ^
  --hidden-import soundfile --hidden-import audioread ^
  --hidden-import matplotlib --hidden-import PIL ^
  Software\src\desktop_app.py
echo BUILD EXIT: %ERRORLEVEL%

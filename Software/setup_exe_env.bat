@echo off
REM Build the isolated CPU-only venv for the exe. Run from VIDS root.
cd /d "%~dp0.."
if not exist "Software\exe_env\Scripts\python.exe" (
  echo [1/4] creating venv...
  python -m venv "Software\exe_env"
)
echo [2/4] upgrading pip...
"Software\exe_env\Scripts\python.exe" -m pip install --upgrade pip
echo [3/4] installing CPU torch (this is large, takes a few minutes)...
"Software\exe_env\Scripts\python.exe" -m pip install torch --index-url https://download.pytorch.org/whl/cpu
echo [4/4] installing app deps + pyinstaller...
"Software\exe_env\Scripts\python.exe" -m pip install flask librosa matplotlib soundfile audioread pywebview numpy scipy pillow pyinstaller
echo [verify]
"Software\exe_env\Scripts\python.exe" -c "import torch; print('venv torch:', torch.__version__, '| cuda:', torch.cuda.is_available())"

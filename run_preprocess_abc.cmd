@echo off
setlocal

rem Run ABC STEP -> UV-Net graph preprocessing (full dataset)
set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%" || exit /b 1

if not exist ".venv_occ\Scripts\python.exe" (
  echo Missing .venv_occ\Scripts\python.exe
  echo Please create the uv-managed venv first.
  exit /b 1
)

set "DGLBACKEND=pytorch"
set "PYTHONWARNINGS=ignore::DeprecationWarning"

".venv_occ\Scripts\python.exe" -m uvnet_retrieval.preprocess_abc ^
  --root "C:\Users\hanyi\workspace\ABC-DataSet\abc_0020_step_v00" ^
  --out_root "C:\Users\hanyi\workspace\ABC-DataSet\abc_0020_step_v00_graphs" ^
  --layout flat ^
  --uvnet_root "external/uvnet" ^
  --python ".venv_occ\Scripts\python.exe"

endlocal

@echo off
rem ptx - unified PyTorchX CLI:  ptx <task> <mode> [args...]
rem Set PTX_PYTHON to force an interpreter, otherwise %CONDA_PREFIX% is used.
setlocal
if defined PTX_PYTHON (set "PY=%PTX_PYTHON%") else if defined CONDA_PREFIX (set "PY=%CONDA_PREFIX%\python.exe") else if exist "C:\ProgramData\miniconda3\envs\ptocr\python.exe" (set "PY=C:\ProgramData\miniconda3\envs\ptocr\python.exe") else (set "PY=python")
"%PY%" -m pytorchx %*

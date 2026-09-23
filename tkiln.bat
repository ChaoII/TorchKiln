@echo off
rem tkiln - unified torchkiln CLI:  tkiln <task> <mode> [args...]
rem Set TKILN_PYTHON to force an interpreter, otherwise %CONDA_PREFIX% is used.
setlocal
if defined TKILN_PYTHON (set "PY=%TKILN_PYTHON%") else if defined CONDA_PREFIX (set "PY=%CONDA_PREFIX%\python.exe") else if exist "C:\ProgramData\miniconda3\envs\ptocr\python.exe" (set "PY=C:\ProgramData\miniconda3\envs\ptocr\python.exe") else (set "PY=python")
"%PY%" -m torchkiln %*

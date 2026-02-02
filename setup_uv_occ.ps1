$ErrorActionPreference = 'Stop'

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

$micromamba = Join-Path $root '.tools\Library\bin\micromamba.exe'
if (-not (Test-Path $micromamba)) {
  Write-Error "Missing $micromamba."
  exit 1
}

$uvExe = (Get-Command uv.exe -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Source -First 1)
if (-not $uvExe) {
  $candidates = @(
    "$env:USERPROFILE\AppData\Roaming\Python\Python311\Scripts\uv.exe",
    "$env:USERPROFILE\AppData\Roaming\Python\Python313t\Scripts\uv.exe"
  )
  $uvExe = $candidates | Where-Object { Test-Path $_ } | Select-Object -First 1
}
if (-not $uvExe) {
  Write-Error 'uv.exe not found. Install with: py -m pip install --user uv'
  exit 1
}

$condaPython = Join-Path $root '.conda\occ\python.exe'
if (-not (Test-Path $condaPython)) {
  & $micromamba create -y -p .conda\occ -c conda-forge python=3.9 pythonocc-core
}

& $uvExe venv -p .conda\occ\python.exe --system-site-packages .venv_occ

& $uvExe pip install -p .venv_occ\Scripts\python.exe --index-url https://download.pytorch.org/whl/cpu torch==2.3.0+cpu
& $uvExe pip install -p .venv_occ\Scripts\python.exe "numpy<2" pyyaml pydantic tqdm torchdata==0.7.1 dgl==2.2.1 deprecate git+https://github.com/AutodeskAILab/occwl

$dataExchange = Join-Path $root '.conda\occ\lib\site-packages\OCC\Extend\DataExchange.py'
if (-not (Select-String -Path $dataExchange -Pattern 'list_of_shapes_to_compound' -Quiet)) {
  Add-Content -Path $dataExchange -Value @"


def list_of_shapes_to_compound(shapes):
    from OCC.Core.TopoDS import TopoDS_Compound
    from OCC.Core.BRep import BRep_Builder
    compound = TopoDS_Compound()
    builder = BRep_Builder()
    builder.MakeCompound(compound)
    for shape in shapes:
        builder.Add(compound, shape)
    return compound, True
"@
}

$deprecateFile = Join-Path $root '.venv_occ\lib\site-packages\deprecate\__init__.py'
if (-not (Select-String -Path $deprecateFile -Pattern 'def deprecated\(func=None' -Quiet)) {
  Set-Content -Path $deprecateFile -Encoding ASCII -Value @"
# -*- coding: utf-8 -*-
import warnings
import functools


def deprecated(func=None, **_kwargs):
    if func is None:
        def decorator(inner):
            return deprecated(inner)
        return decorator
    @functools.wraps(func)
    def new_func(*args, **kwargs):
        code = func.func_code if hasattr(func, "func_code") else func.__code__
        warnings.warn_explicit(
            "Call to deprecated function {}.".format(func.__name__),
            category=DeprecationWarning,
            filename=code.co_filename,
            lineno=code.co_firstlineno + 1
        )
        return func(*args, **kwargs)
    return new_func
"@
}

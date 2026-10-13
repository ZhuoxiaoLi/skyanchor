from pathlib import Path

from .gvars import ROOT

SMERF_DATA_DIR = Path(ROOT, "local/benchmarks/smerf/data")
SMERF_CACHE_DIR = Path(ROOT, "local/benchmarks/smerf/cache_dir")
SMERF_EXP_DIR = Path(ROOT, "local/benchmarks/smerf/experiments")

ETH3D_DATA_DIR = Path(ROOT, "local/benchmarks/eth3d/data")
ETH3D_CACHE_DIR = Path(ROOT, "local/benchmarks/eth3d/cache_dir")
ETH3D_EXP_DIR = Path(ROOT, "local/benchmarks/eth3d/experiments")

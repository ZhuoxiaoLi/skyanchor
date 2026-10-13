from pathlib import Path

MPSFM_DIR = Path(__file__).parent.parent
ROOT = MPSFM_DIR.parent

SFM_CONFIG_DIR = ROOT / "configs"
MONO_MODEL_CONFIG_DIR = MPSFM_DIR / "extraction/imagewise/geometry/models/configs"

TESTSETS_DIR = ROOT / "local/testsets"

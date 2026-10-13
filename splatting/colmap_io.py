"""Use the same COLMAP serialization as the pose reconstruction stage."""
import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    'skyanchor_colmap_io',
    Path(__file__).resolve().parents[1] / 'anchorRecon/dataloaders/read_write_model.py',
)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
read_model = _module.read_model
write_model = _module.write_model

from pathlib import Path
import sys
from urllib.parse import urlparse

import torch


def is_loma_backend(name: str) -> bool:
    return name.lower().replace("_", "-") in {
        "loma",
        "lomag",
        "loma-g",
        "loma-b",
        "loma-b128",
        "loma-l",
    }


def _import_loma():
    try:
        from loma import LoMa, LoMaB, LoMaB128, LoMaG, LoMaL
        from loma.loma import filter_matches, to_pixel_coords
        return LoMa, LoMaB, LoMaB128, LoMaG, LoMaL, filter_matches, to_pixel_coords
    except ImportError:
        repo_root = Path(__file__).resolve().parents[1]
        local_src = repo_root / "submodules" / "LoMa" / "src"
        if local_src.exists():
            sys.path.insert(0, str(local_src))
        from loma import LoMa, LoMaB, LoMaB128, LoMaG, LoMaL
        from loma.loma import filter_matches, to_pixel_coords
        return LoMa, LoMaB, LoMaB128, LoMaG, LoMaL, filter_matches, to_pixel_coords


def build_loma_model(
    backend: str,
    num_keypoints: int,
    filter_threshold: float = -1.0,
    compile_model: bool = False,
):
    LoMa, LoMaB, LoMaB128, LoMaG, LoMaL, _, _ = _import_loma()
    backend = backend.lower().replace("_", "-")
    cfg_cls = {
        "loma": LoMaG,
        "lomag": LoMaG,
        "loma-g": LoMaG,
        "loma-b": LoMaB,
        "loma-b128": LoMaB128,
        "loma-l": LoMaL,
    }[backend]
    cfg_kwargs = {
        "num_keypoints": num_keypoints,
        "compile": compile_model,
    }
    if filter_threshold >= 0:
        cfg_kwargs["filter_threshold"] = filter_threshold
    cfg = cfg_cls(**cfg_kwargs)
    try:
        return LoMa(cfg)
    except RuntimeError as exc:
        if "PytorchStreamReader" not in str(exc):
            raise
        cached_path = _torch_hub_checkpoint_path(cfg.weights_url)
        if cached_path is None or not cached_path.exists():
            raise
        print(f"Removing corrupt LoMa checkpoint cache: {cached_path}")
        cached_path.unlink()
        return LoMa(cfg)


def _torch_hub_checkpoint_path(url: str | None):
    if not url:
        return None
    filename = Path(urlparse(url).path).name
    if not filename:
        return None
    return Path(torch.hub.get_dir()) / "checkpoints" / filename


def get_loma_match_utils():
    *_, filter_matches, to_pixel_coords = _import_loma()
    return filter_matches, to_pixel_coords

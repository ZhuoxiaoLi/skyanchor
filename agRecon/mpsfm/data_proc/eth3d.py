"""ETH3D dataset parser and dataset class."""

import torch

from mpsfm.data_proc.basedataset import BaseDataset, BaseDatasetParser
from mpsfm.vars import gvars, lvars


class ETH3DDataset(BaseDataset, torch.utils.data.Dataset):
    """Dataset class for ETH3D dataset."""

    data_dir = lvars.ETH3D_DATA_DIR
    default_exp_dir = lvars.ETH3D_EXP_DIR
    default_cache_dir = lvars.ETH3D_CACHE_DIR
    testsets = gvars.TESTSETS_DIR / "eth3d"
    scenes = [el.name for el in testsets.iterdir()]


class ETH3DParser(BaseDatasetParser):
    """Parser for ETH3D dataset."""

    dataset = ETH3DDataset

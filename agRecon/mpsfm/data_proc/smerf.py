"""SMERF dataset and parser for MP-SfM pipeline."""

import torch

from mpsfm.vars import gvars, lvars

from .basedataset import BaseDataset, BaseDatasetParser


class SMERFDataset(BaseDataset, torch.utils.data.Dataset):
    """Dataset class for SMERF data."""

    data_dir = lvars.SMERF_DATA_DIR
    default_exp_dir = lvars.SMERF_EXP_DIR
    default_cache_dir = lvars.SMERF_CACHE_DIR
    testsets = gvars.TESTSETS_DIR / "smerf"
    scenes = [el.name for el in testsets.iterdir()]


class SMERFParser(BaseDatasetParser):
    """Parser for SMERF dataset."""

    dataset = SMERFDataset

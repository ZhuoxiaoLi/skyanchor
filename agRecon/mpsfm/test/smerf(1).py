from mpsfm.data_proc.prepare.smerf import main as smerf_download
from mpsfm.data_proc.smerf import SMERFDataset, SMERFParser

from .basetest import BaseTest


class SMERFTest(BaseTest):
    dataset = SMERFDataset
    parser = SMERFParser

    def _auto_download(self):
        """Execute the auto-download function for the ETH3D dataset."""
        print("Downloading and processing SMERF dataset...")
        smerf_download()

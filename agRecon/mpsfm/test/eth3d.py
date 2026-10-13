from mpsfm.data_proc.eth3d import ETH3DDataset, ETH3DParser
from mpsfm.data_proc.prepare.eth3d import main as eth3d_download

from .basetest import BaseTest


class ETH3DTest(BaseTest):
    """ETH3D Test class for evaluating reconstruction performance on ETH3D testsets."""

    dataset = ETH3DDataset
    parser = ETH3DParser

    def _auto_download(self):
        """Execute the auto-download function for the ETH3D dataset."""
        print("Downloading and processing ETH3D dataset...")
        eth3d_download()

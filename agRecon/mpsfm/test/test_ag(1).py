from pathlib import Path

from mpsfm.baseclass import BaseClass
from mpsfm.data_proc.AGData import AGDataset, AGParser
from mpsfm.sfm.reconstruction_AGmanager import AGReconstructionManager

class AGTest(BaseClass):
    """SimpleTest class for reconstructing a user input testset."""

    dataset = AGDataset
    default_conf = {"dataset": {"name": "simple"}}
    freeze_conf = False

    def __call__(
        self,
        data_output_dir=None,
        intrinsics_pth=None,
        recon_exist_dir=None,
        cache_dir=None,
        rgb_register_dir=None,
        rgb_unregister_dir=None,
        submap_im_num=50,
        anchor_num_pair=(6,6),
    ):
        data_output_dir = Path(data_output_dir)

        scene_parser = AGParser(
            intrinsics_pth=intrinsics_pth,
            recon_exist_dir=recon_exist_dir,
            rgb_unregister_dir=rgb_unregister_dir,
            rgb_register_dir=rgb_register_dir,
            submap_im_num=submap_im_num,
            anchor_num_pair=anchor_num_pair,
        )

        if cache_dir is None:
            cache_dir = data_output_dir / "cache_dir"

        self.reconstruction_manager = AGReconstructionManager(self.conf)
        init_info = dict(
            references=scene_parser.imnames,
            sfm_outputs_dir=Path(data_output_dir) / "sfm_outputs",
            cache_dir=cache_dir,
            ref_imids=list(scene_parser.rec.images.keys()),
            scene_parser=scene_parser,
        )
        mpsfm_rec, sfm_rec = self.reconstruction_manager(**init_info)
        return sfm_rec

import logging
import torch

from mpsfm.extraction.base_model import BaseModel

from .utils.warp import (
    assign_keypoints,
    kpids_to_matches0,
    simple_nms,
)

from romav2 import RoMaV2

logger = logging.getLogger(__name__)

class Romav2(BaseModel):
    default_conf = {
        "model_name": "roma_outdoor.pth",
        "model_utils_name": "dinov2_vitl14_pretrain.pth",
        "sample_thresh": 0.1,
        "nms_radius": 8,
        "max_error": 2,
        "require_download": False,
        "sample_num": 5000
    }
    required_inputs = [
        "image0",
        "image1",
    ]

    # Initialize the line matcher
    def _init(self, conf):
        logger.info("Loading Romav2 model...")
        # load the model

        self.net = RoMaV2()
        self.net.apply_setting("precise")
        self.net.eval()
        self.conf = conf
        logger.info("Load Romav2 model done.")

    def _forward(self, data, mode="sparse"):
        img0 = data["image0"]
        img1 = data["image1"]
        H_A, W_A = img0.shape[-2], img0.shape[-1]
        H_B, W_B = img1.shape[-2], img1.shape[-1]
        # imgdir0 = data["imdir0"]
        # imgdir1 = data["imdir1"]

        with torch.inference_mode():
            preds = self.net.match(img0, img1)

        warp = preds["warp_AB"][0]
        certainty = preds["overlap_AB"][0].squeeze(-1)

        H, W, _ = warp.shape
        grid = self.get_normalized_grid_hw(H, W, device=warp.device, dtype=warp.dtype)
        matches = torch.cat([grid, warp], dim=-1).reshape(-1, 4)  # (H*W, 4)


        if "dense" in mode:
            #matches_nms, certainty_nms, precision_AB, precision_BA = self.net.sample(preds, int(self.conf["sample_num"]))

            certainty_nms = simple_nms(certainty, self.conf["nms_radius"])
            certainty_nms = certainty_nms.reshape(-1)
            mask = certainty_nms > self.conf["sample_thresh"]
            certainty_nms = certainty_nms[mask]
            matches_nms = matches[mask]

            dkps0, dkps1 = self.net.to_pixel_coordinates(matches_nms, H_A, W_A, H_B, W_B)
            dkps0 = dkps0.cpu().numpy()
            dkps1 = dkps1.cpu().numpy()
            scores0 = certainty_nms.cpu().numpy()
        if "sparse" in mode:
            certainty = certainty.reshape(-1).cpu().numpy()
            kpts0, kpts1 = self.net.to_pixel_coordinates(matches, H_A, W_A, H_B, W_B)
            kpts0 = kpts0.cpu().numpy()
            kpts1 = kpts1.cpu().numpy()
            skpts0 = data["skpts0"]
            skpts1 = data["skpts1"]
            mkp_ids0 = assign_keypoints(kpts0 * data["scale0"], skpts0, self.conf.max_error)
            mkp_ids1 = assign_keypoints(kpts1 * data["scale1"], skpts1, self.conf.max_error)
            smatches, sscores = kpids_to_matches0(mkp_ids0, mkp_ids1, certainty)

        pred = {}
        if "dense" in mode:
            pred["dkeypoints0"], pred["dkeypoints1"] = dkps0, dkps1
            pred["dscores"] = scores0
        if "sparse" in mode:
            pred["smatches0"] = smatches
            pred["smatching_scores0"] = sscores

        return pred

    @staticmethod
    def get_normalized_grid_hw(H: int, W: int, device: torch.device, dtype: torch.dtype | None = None) -> torch.Tensor:
        """
        Return grid with shape (H, W, 2), last dim is (x, y),
        on the given device (and dtype if provided).
        """
        dtype = dtype or torch.float32
        y = torch.linspace(-1 + 1 / H, 1 - 1 / H, H, device=device, dtype=dtype)
        x = torch.linspace(-1 + 1 / W, 1 - 1 / W, W, device=device, dtype=dtype)

        yy, xx = torch.meshgrid(y, x, indexing="ij")  # (H, W)
        grid = torch.stack((xx, yy), dim=-1)  # (H, W, 2)
        return grid

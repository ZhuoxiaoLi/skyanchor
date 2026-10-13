# vpr_extractor.py
import torch

from mpsfm.extraction.base_model import BaseModel

class AnyLocExtractor(BaseModel):
    default_conf = {
        "device": "cuda",
        "require_download": False,
    }
    required_inputs = ["image"]

    def _init(self, conf):
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = torch.hub.load("AnyLoc/DINO", "get_vlad_model", domain="urban", backbone="DINOv2", device="cuda")
        self.net.eval()
        self.net.cuda()

    @torch.inference_mode()
    def _forward(self, data):
        """Read data["image"] as float32 tensors (B, C, H, W) and return {"descriptors": tensor} with shape (B, D)."""
        images = data["image"]

        desc = self.net(images)

        return {"global_descriptor": desc}

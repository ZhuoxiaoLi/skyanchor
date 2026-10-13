# vpr_extractor.py
import torch

from mpsfm.extraction.base_model import BaseModel

class MegaLocExtractor(BaseModel):
    default_conf = {
        "method": "cosplace",
        "backbone": "resnet50",
        "descriptors_dimension": 512,
        "device": "cuda",
        "require_download": False,
    }
    required_inputs = ["image"]

    def _init(self, conf):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = torch.hub.load("gmberton/MegaLoc", "get_trained_model")
        self.net = self.net.eval().to(device)

    @torch.inference_mode()
    def _forward(self, data):
        """Read data["image"] as float32 tensors (B, C, H, W) and return {"descriptors": tensor} with shape (B, D)."""
        images = data["image"]

        desc = self.net(images)

        return {"global_descriptor": desc}

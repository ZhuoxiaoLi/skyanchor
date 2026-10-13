import argparse
from pathlib import Path
import time

from mpsfm.test.test_ag import AGTest
from mpsfm.utils.tools import load_cfg
from mpsfm.vars import gvars

parser = argparse.ArgumentParser()
parser.add_argument(
    "--data_output_dir", type=str, default="local/example", help="Main data dir storing later the outputs"
)
parser.add_argument("--rgb_register_dir", type=str, help="register Images directory")
parser.add_argument("--rgb_unregister_dir", type=str, help="unregister Images directory")

parser.add_argument("--intrinsics_pth", type=str, default=None, help="Path to intrinsics .yaml file, for ground images")
parser.add_argument("--recon_exist_dir", type=str, default=None, help="Path to reference reconstruction, aerial reconstruction")
parser.add_argument("--cache_dir", type=str, default=None, help="Path to cache directory")
parser.add_argument("-e", "--extract", nargs="*", type=str, default=[], help="List of priors to force reextract")
parser.add_argument("-c", "--conf", type=str, help="Name of the sfm config file", default="sp-mast3r-dense")
parser.add_argument("-v", "--verbose", type=int, default=0)

parser.add_argument("--submap_im_num", type=int, default=50, help="image number of submap")
parser.add_argument("--anchor_num_pair", type=int, nargs=2, default=(6,6), help="the number of anchor images in a submap")

start_time = time.time()
args, _ = parser.parse_known_args()
conf = load_cfg(gvars.SFM_CONFIG_DIR / f"{args.conf}.yaml", return_name=False)
conf.extract = args.extract
conf.verbose = args.verbose

experiment = AGTest(conf)

mpsfm_rec = experiment(
    data_output_dir=args.data_output_dir,
    intrinsics_pth=args.intrinsics_pth,
    recon_exist_dir=args.recon_exist_dir,
    cache_dir=args.cache_dir,
    rgb_register_dir=args.rgb_register_dir,
    rgb_unregister_dir=args.rgb_unregister_dir,
    submap_im_num=args.submap_im_num,
    anchor_num_pair=args.anchor_num_pair,
)
sfm_outputs_dir = Path(args.data_output_dir) / "sfm_outputs"
sfm_outputs_dir.mkdir(exist_ok=True)
rec_dir = sfm_outputs_dir / "rec"
rec_dir.mkdir(parents=True, exist_ok=True)
mpsfm_rec.write(rec_dir)

end_time = time.time()
elapsed_time = end_time - start_time

hours, remainder = divmod(elapsed_time, 3600)
minutes, seconds = divmod(remainder, 60)
print(f"Total runtime: {int(hours)}h {int(minutes)}m {seconds:.2f}s")

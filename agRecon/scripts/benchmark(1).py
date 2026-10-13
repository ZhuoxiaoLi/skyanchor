import argparse

# import omegaconfig
from omegaconf import OmegaConf

from mpsfm.test import get_test
from mpsfm.utils.tools import load_cfg
from mpsfm.vars import gvars

parser = argparse.ArgumentParser()
parser.add_argument(
    "-d", "--dataset", choices=["eth3d", "smerf"], default="eth3d", help="Dataset to use: 'eth3d' or 'smerf'."
)
parser.add_argument(
    "-c", "--conf", type=str, default="paper/repr-sp-lg_m3dv2", help="Config file name in SFM_CONFIG_DIR"
)
parser.add_argument("-t", "--terminate", action="store_true", help="Terminate if error is caught")
parser.add_argument("-e", "--extract", nargs="+", type=str, help="Extract needed priors", default=[])
parser.add_argument("--testset_id", type=int, help="Testset id to run")
parser.add_argument("-v", "--verbose", type=int, default=0)
parser.add_argument("-o", "--overwrite", action="store_true", help="Overwrite existing results")
parser.add_argument("-s", "--scene", type=str)
parser.add_argument("-m", "--mode", type=str, default="minimal")

args, _ = parser.parse_known_args()
conf = load_cfg(gvars.SFM_CONFIG_DIR / f"{args.conf}.yaml")
conf = OmegaConf.create(conf)

experiment = get_test(args.dataset).init_with_parser(conf, args)
experiment()

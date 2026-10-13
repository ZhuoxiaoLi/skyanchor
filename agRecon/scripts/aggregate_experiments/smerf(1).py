import contextlib
from argparse import ArgumentParser

from tqdm import tqdm

from mpsfm.eval.sfm.dataset_aggregators.smerf import SMERFAggregator

parser = ArgumentParser()
parser.add_argument("-c", "--configs", nargs="+", help="Configs to aggregate", default=["repr-sp-lg_m3dv2"])
parser.add_argument(
    "-m",
    "--modes",
    nargs="+",
    help="Configs to aggregate",
    default=["minimal", "low", "medium", "high"],
)
args = parser.parse_args()
out = {}
for mode in tqdm(args.modes):
    agg = SMERFAggregator(sfm_configs=args.configs, mode=mode)

    out[mode] = agg.all_scenes()
print("Modes:".ljust(19), "     ".join(args.modes))
for conf in args.configs:
    with contextlib.suppress(Exception):
        print(f"{conf.ljust(15)}     {'     '.join([out[mode][conf] for mode in args.modes])}")

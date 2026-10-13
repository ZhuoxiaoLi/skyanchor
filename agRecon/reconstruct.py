import argparse
from pathlib import Path

from mpsfm.test.simple import SimpleTest
from mpsfm.utils.tools import load_cfg
from mpsfm.vars import gvars

import time

import psutil
import threading
import os

def bytes_to_gb(x: float) -> float:
    return x / 1024 / 1024 / 1024


class MemoryMonitor:
    """Monitor resident memory of this process and its children. Track maximum RSS and the latest RSS sample."""

    def __init__(self, interval: float = 1.0):
        self.interval = interval
        self.process = psutil.Process(os.getpid())
        self.max_rss = 0
        self.last_rss = 0
        self.running = False
        self.thread = None

    def _get_total_rss(self) -> int:
        total = 0

        try:
            total += self.process.memory_info().rss
        except psutil.NoSuchProcess:
            pass

        for child in self.process.children(recursive=True):
            try:
                total += child.memory_info().rss
            except psutil.NoSuchProcess:
                pass

        return total

    def _monitor(self):
        while self.running:
            rss = self._get_total_rss()
            self.last_rss = rss
            self.max_rss = max(self.max_rss, rss)
            time.sleep(self.interval)

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._monitor, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread is not None:
            self.thread.join()

parser = argparse.ArgumentParser()
parser.add_argument(
    "--data_dir", type=str, default="local/example", help="Main data dir storing inputs and later the outputs"
)
parser.add_argument("--images_dir", type=str, help="Images directory")
parser.add_argument(
    "--imnames", type=str, nargs="*", help="List of image names to process. Leave empty to process all images"
)
parser.add_argument("--intrinsics_pth", type=str, default=None, help="Path to intrinsics .yaml file")
parser.add_argument("--refrec_dir", type=str, default=None, help="Path to reference reconstruction")
parser.add_argument("--cache_dir", type=str, default=None, help="Path to cache directory")
parser.add_argument("-e", "--extract", nargs="*", type=str, default=[], help="List of priors to force reextract")
parser.add_argument("-c", "--conf", type=str, help="Name of the sfm config file", default="sp-lg_m3dv2")
parser.add_argument("-v", "--verbose", type=int, default=0)

start_time = time.time()

mem_monitor = MemoryMonitor(interval=1.0)
mem_monitor.start()

try:
    args, _ = parser.parse_known_args()

    conf = load_cfg(
        gvars.SFM_CONFIG_DIR / f"{args.conf}.yaml",
        return_name=False
    )
    conf.extract = args.extract
    conf.verbose = args.verbose

    experiment = SimpleTest(conf)

    mpsfm_rec = experiment(
        imnames=args.imnames,
        intrinsics_pth=args.intrinsics_pth,
        refrec_dir=args.refrec_dir,
        cache_dir=args.cache_dir,
        data_dir=args.data_dir,
        images_dir=args.images_dir,
    )

    sfm_outputs_dir = Path(args.data_dir) / "sfm_outputs"
    sfm_outputs_dir.mkdir(exist_ok=True)

    mpsfm_rec.write(sfm_outputs_dir)

finally:
    mem_monitor.stop()

end_time = time.time()
elapsed_time = end_time - start_time

hours, remainder = divmod(elapsed_time, 3600)
minutes, seconds = divmod(remainder, 60)

print(f"Total runtime: {int(hours)}h {int(minutes)}m {seconds:.2f}s")
print(f"Final RSS memory: {bytes_to_gb(mem_monitor.last_rss):.3f} GB")
print(f"Peak RSS memory:  {bytes_to_gb(mem_monitor.max_rss):.3f} GB")

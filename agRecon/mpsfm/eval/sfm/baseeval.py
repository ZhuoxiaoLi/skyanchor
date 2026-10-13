from abc import ABCMeta, abstractmethod
from pathlib import Path

import numpy as np
import yaml
from omegaconf import OmegaConf


def convert_numpy(obj):
    if isinstance(obj, np.generic):
        return obj.item()
    elif isinstance(obj, dict):
        return {k: convert_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy(v) for v in obj]
    else:
        return obj


class BaseAggregator(metaclass=ABCMeta):
    base_default_conf = {}
    default_conf = {}

    def __init__(self, conf):
        default_conf = OmegaConf.merge(
            OmegaConf.create(self.base_default_conf),
            OmegaConf.create(self.default_conf),
        )
        if isinstance(conf, dict):
            conf = OmegaConf.create(conf)
        self.conf = OmegaConf.merge(default_conf, conf)
        self.aggr_desc = None

    def _setup(
        self,
        path_template,
        exp_dir=None,
        scene_desc=None,
        group_desc=None,
        conf=None,
        aggr_desc=None,
        recdescs=None,
        verbose=1,
        specific=False,
    ):
        # print("Setting up aggregator...")
        self.path_template = (
            Path(
                path_template.format(
                    exp_dir=exp_dir, scene=scene_desc, testset_type=group_desc, conf=conf, testset_desc="*"
                )
            )
            / "results"
        )
        if aggr_desc is None:
            self.aggr_desc = f"Scene: {scene_desc} Group: {group_desc} Conf: {conf}"
        else:
            self.aggr_desc = aggr_desc
        part_ids = [i for i, part in enumerate(self.path_template.parts) if "*" in part]
        self.aggregated_evals = {}
        count_direcoties_not_found = 0
        if recdescs is None:
            eval_dirs = list(self.find_matching_paths(self.path_template))
        else:
            tot_exp_recs = sum([len(el) for el in recdescs.values()])
            eval_dirs = []
            count = 0

            for scene in recdescs if not specific else [scene_desc]:
                self.path_template = (
                    Path(
                        path_template.format(
                            exp_dir=exp_dir, scene=scene, testset_type=group_desc, conf=conf, testset_desc="*"
                        )
                    )
                    / "results"
                )
                if len(list(self.find_matching_paths(self.path_template))) == 0:
                    count_direcoties_not_found += 1
                    if verbose > 1:
                        print(f"No matching directories found for {self.path_template}")
                    continue
                count += 1
                eval_dirs += list(self.find_matching_paths(self.path_template, ids=recdescs[scene]))
            if (count != len(recdescs) or len(eval_dirs) != tot_exp_recs) and verbose:
                print(f"Could not find all directories for {path_template}")
                print(f"Found reconstrucitons of {count} scenes w a total of {len(eval_dirs)} recs")
                print(f"Expected reconstructions of {len(recdescs)} scenes w a total of {tot_exp_recs} recs")

        if verbose and count_direcoties_not_found > 0:
            print(50 * "-")
            print(f"Could not find {count_direcoties_not_found} directories")
            print(50 * "-")
        if len(eval_dirs) == 0:
            print(f"No matching directories found for {self.path_template}")
            return False
        count_results_not_found = 0
        for eval_dir in eval_dirs:
            results_name = "-".join(
                [eval_dir.parts[i - 1 + len(Path(list(recdescs.keys())[0]).parts)] for i in part_ids]
            )
            eval_obj = self.eval_obj()
            verbose = 2
            if not eval_obj.load(eval_dir, verbose=verbose):
                if verbose > 1:
                    print(f"Could not load resutls for {results_name}: {eval_dir}")
                count_results_not_found += 1
                continue
            if not eval_obj.valid():
                print(f"Results for {results_name} are invalid: {eval_dir}")
                continue
            # check if eval_obj has .conf
            if not hasattr(eval_obj, "conf"):
                print(f"Results for {results_name} do not have a conf: {eval_dir}")
                continue
            self.aggregated_evals[results_name] = eval_obj
        if verbose and count_results_not_found > 0:
            print(50 * "-")
            print(f"Could not find {count_results_not_found} result files")
            print(50 * "-")
        return self.aggregated_evals

    @staticmethod
    def find_matching_paths(path_template, ids=None):
        def _find_matching_paths(current_path, pattern_parts, depth=0):
            if depth == len(pattern_parts):
                if current_path.exists():
                    if ids is not None:
                        split = current_path.parent.parent.name.split("_")
                        if len(split) != 1:
                            if set([int(el) for el in split]) not in ids:
                                return
                        else:
                            if int(split[0]) not in ids:
                                return
                    yield current_path
                return
            current_part = pattern_parts[depth]

            if "*" in current_part:
                if current_part == "**":
                    for pth in current_path.rglob("*"):
                        yield from _find_matching_paths(pth, pattern_parts, depth + 1)
                else:
                    for pth in current_path.glob(current_part):  # Handle single '*' wildcard
                        yield from _find_matching_paths(pth, pattern_parts, depth + 1)
            else:
                next_path = current_path / current_part
                yield from _find_matching_paths(next_path, pattern_parts, depth + 1)

        pattern_parts = path_template.parts
        return _find_matching_paths(Path(), pattern_parts)

    @abstractmethod
    def aggregate(self):
        """To be implemented by the child class."""

    def summarize(self, **kwargs):
        summary = self._summarize(**kwargs)
        return summary


class BaseEval(metaclass=ABCMeta):
    base_default_conf = {}
    default_conf = {}

    def __init__(self, conf=None):
        self.conf = OmegaConf.merge(
            OmegaConf.create(self.base_default_conf),
            OmegaConf.create(self.default_conf),
        )
        if conf is not None:
            if isinstance(conf, dict):
                conf = OmegaConf.create(conf)
            self.conf = OmegaConf.merge(self.conf, conf)

    def setup(self, estim_rec, gt_rec):
        for imid in estim_rec.images:
            assert imid in gt_rec.images, f"Image {imid} not in ground truth reconstruction"
        self.estim_rec = estim_rec
        self.gt_rec = gt_rec

        self.results = {
            "summary": {},
            "results": {},
            "full_results": None,
            "num_images": self.estim_rec.num_images(),
            "num_registered_images": self.estim_rec.num_reg_images(),
            "success": False,
            "conf": OmegaConf.to_container(self.conf, resolve=True),
        }

    @abstractmethod
    def compute(self, **kwargs):
        """To be implemented by the child class."""
        raise NotImplementedError

    @abstractmethod
    def _summarize(self):
        """To be implemented by the child class."""
        raise NotImplementedError

    def summarize(self, **kwargs):
        summary = self._summarize(**kwargs)
        assert isinstance(summary, str), "Summary must be a string"
        print(summary)

    def save_results(self, results_dir):
        self.write_full_results(results_dir / "full_results.yaml")
        self.write_summary(results_dir / "summary.txt")

    def write_summary(self, fname):
        """To be implemented by the child class."""
        summary = self._summarize()
        assert isinstance(summary, str), "Summary must be a string"
        with open(fname, "w") as file:
            file.write(summary)

    def load(self, results_dir, verbose=1):
        """Load results from a directory."""
        self.results_dir = results_dir
        fname = results_dir / "full_results.yaml"
        if not fname.exists():
            if verbose > 1:
                print(f"Results file {fname} does not exist")
            return False
        with open(fname) as file:
            self.results = yaml.load(file, Loader=yaml.FullLoader)
            self.conf = OmegaConf.create(self.results["conf"])
        return True

    def valid(self):
        """Check if the results are valid."""
        return self.results["success"]

    def write_full_results(self, fname):
        """Write the full results to a file."""
        with open(fname, "w") as file:
            yaml.dump(convert_numpy(dict(self.results)), file, default_flow_style=False)

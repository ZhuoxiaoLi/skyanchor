from pathlib import Path

import torch
from omegaconf import OmegaConf

from mpsfm.baseclass import BaseClass
from mpsfm.utils.tools import load_cfg
from mpsfm.vars import gvars

from .imagewise import features as extract_features
from .imagewise import geometry as extract_mono
from .imagewise import mask as extract_skyseg
from .pairs import pairs_from_exhaustive, pairs_from_retrieval, pairs_from_sequential, FolderKDIndex, build_image_kdtree, query_nearest_images_by_names_from_kdtree
from .pairwise import CONFIG_DIR as PAIRWISE_CONFIG_DIR
from .pairwise import match_dense_2view
from .pairwise import match_sparse as match_features


# Monkey patch for smooth extraction.
_real_torch_load = torch.load


def safe_load(*args, **kwargs):
    if "weights_only" not in kwargs:
        kwargs["weights_only"] = False
    return _real_torch_load(*args, **kwargs)


torch.load = safe_load


class Extraction(BaseClass):
    """Extraction class for the MP-SfM pipeline."""

    default_conf = {
        # depth priors
        "depth": "metric3dv2",
        "normals": "metric3dv2",
        # features
        "features": "superpoint",
        "matcher": "superpoint+lightglue",
        #   sparse-dense
        "sparse_nms_radius": 6,
        # pairs
        "retrieval": "AnyLoc", # netvlad MegaLoc Salad AnyLoc
        "nquery": 20,  # retrieval
        "nquery_loca": 3,
        "noverlap": 2,  # sequential
        "quadratic_overlap": False,  # sequential
        "dataset": {
            "name": "simple",
        },
        "matches_mode": None,
        "verbose": 0,
        "initial_recon_image_num": 8,
        "internal_image_num": 20,
        "nails_num": 3,
    }

    def __init__(
        self,
        conf,
        models=None,
        cache_dir=None,
        sfm_outputs_dir=None,
        scene_parser=None,
        references=None,
        extract=None,
        i=None
    ):
        if isinstance(conf, dict):
            conf = OmegaConf.create(conf)
        self.conf = OmegaConf.merge(self.default_conf, conf)
        if models is None:
            models = {}
        self.models = models

        self.scene_parser = scene_parser
        if references is None:
            references = [x.stem for x in scene_parser.rgb_dir.glob("*")]
        self.images_list = references
        self.cache_dir = cache_dir
        self.sfm_outputs_dir = sfm_outputs_dir

        self.match_dirs = {}
        self.depth_dir = None
        self.normals_dir = None
        self.retrieval_path = None
        self.sfm_pairs_path = None
        self.sfm_pairs_paths = []
        self.skyseg_dir = None

        self.masks_dirs = []
        if extract is None:
            extract = set()
        self.extract = extract

        self.image_inpairs = []
        self.sfm_pairs_paths_loca = []

    def extract_sparse(self, overwrite=False):
        """Extract sparse features."""
        if any(s in self.extract for s in ["f", "features"]):
            overwrite = True
        feature_conf = load_cfg(extract_features.CONFIG_DIR / f"{self.conf.features}.yaml")
        self.match_dirs["sfeats"], model = extract_features.main(
            feature_conf,
            self.scene_parser,
            self.cache_dir,
            overwrite=overwrite,
            image_list=self.images_list,
            model=self.models.get(self.conf.features, None),
            verbose=self.conf.verbose,
        )
        self.models[self.conf.features] = model
        self.log(f"Features located in {self.match_dirs['sfeats']}", level=1)
        return feature_conf

    def extract_sparse_ag(self, overwrite=False):
        """Extract sparse features."""
        if any(s in self.extract for s in ["f", "features"]):
            overwrite = True
        feature_conf = load_cfg(extract_features.CONFIG_DIR / f"{self.conf.features}.yaml")
        self.match_dirs["sfeats"], model = extract_features.main_ag(
            feature_conf,
            self.scene_parser,
            self.cache_dir,
            overwrite=overwrite,
            image_list=self.images_list,
            model=self.models.get(self.conf.features, None),
            verbose=self.conf.verbose,
        )
        self.models[self.conf.features] = model
        self.log(f"Features located in {self.match_dirs['sfeats']}", level=1)
        return feature_conf

    def match_sparse(self, overwrite=False):
        """Match sparse features."""
        if any(s in self.extract for s in ["m", "matches", "f", "features"]):
            overwrite = True
        matcher_conf = load_cfg(PAIRWISE_CONFIG_DIR / f"{self.conf.matcher}.yaml")
        self.match_dirs["smatches"], model = match_features.main(
            matcher_conf,
            self.sfm_pairs_path,
            self.match_dirs["sfeats"].stem,
            self.cache_dir,
            overwrite=overwrite,
            model=self.models.get(self.conf.matcher, None),
            verbose=self.conf.verbose,
        )

        self.models[self.conf.matcher] = model
        self.log(f"Matches located in {self.match_dirs['smatches']}", level=1)
        return matcher_conf

    def match_sparse_ag(self, overwrite=False, i=None):
        """Match sparse features."""
        if any(s in self.extract for s in ["m", "matches", "f", "features"]):
            overwrite = True
        matcher_conf = load_cfg(PAIRWISE_CONFIG_DIR / f"{self.conf.matcher}.yaml")

        if i is not None:
            self.match_dirs["smatches"], model = match_features.main(
                matcher_conf,
                self.sfm_pairs_paths[i],
                self.match_dirs["sfeats"].stem,
                self.cache_dir,
                overwrite=overwrite,
                model=self.models.get(self.conf.matcher, None),
                verbose=self.conf.verbose,
            )
        else:
            self.match_dirs["smatches"], model = match_features.main(
                matcher_conf,
                self.sfm_pairs_path,
                self.match_dirs["sfeats"].stem,
                self.cache_dir,
                overwrite=overwrite,
                model=self.models.get(self.conf.matcher, None),
                verbose=self.conf.verbose,
            )

        self.models[self.conf.matcher] = model
        self.log(f"Matches located in {self.match_dirs['smatches']}", level=1)
        return matcher_conf

    def match_dense(self, overwrite=False):
        """Match sparse features or extract dense correspondences with a dense matcher."""
        if any(s in self.extract for s in ["m", "matches"]):
            overwrite = True
        matcher_conf = load_cfg(PAIRWISE_CONFIG_DIR / f"{self.conf.matcher}.yaml")
        self.match_dirs, model = match_dense_2view.main(
            matcher_conf,
            self.sfm_pairs_path,
            self.scene_parser,
            self.match_dirs,
            export_dir=self.cache_dir,
            overwrite=overwrite,
            model=self.models.get(self.conf.matcher, None),
            matches_mode=self.conf.matches_mode,
            sparse_nms_radius=self.conf.sparse_nms_radius,
            verbose=self.conf.verbose,
        )
        if "depth" in self.match_dirs:
            self.depth_dir = self.match_dirs["depth"]
        self.models[self.conf.matcher] = model
        if "sparse" in self.conf.matches_mode:
            self.log(f"Matches located in {self.match_dirs['smatches']}", level=1)
        if "dense" in self.conf.matches_mode:
            self.log(f"Matches located in {self.match_dirs['dmatches']}", level=1)
        return matcher_conf

    def match_dense_ag(self, overwrite=False, i=None):
        """Match sparse features or extract dense correspondences with a dense matcher."""
        if any(s in self.extract for s in ["m", "matches"]):
            overwrite = True
        matcher_conf = load_cfg(PAIRWISE_CONFIG_DIR / f"{self.conf.matcher}.yaml")
        if i is not None:
            self.match_dirs, model = match_dense_2view.main_ag(
                matcher_conf,
                self.sfm_pairs_paths[i],
                self.scene_parser,
                self.match_dirs,
                export_dir=self.cache_dir,
                overwrite=overwrite,
                model=self.models.get(self.conf.matcher, None),
                matches_mode=self.conf.matches_mode,
                sparse_nms_radius=self.conf.sparse_nms_radius,
                verbose=self.conf.verbose,
                i=i,
            )
        else:
            self.match_dirs, model = match_dense_2view.main_ag(
                matcher_conf,
                self.sfm_pairs_path,
                self.scene_parser,
                self.match_dirs,
                export_dir=self.cache_dir,
                overwrite=overwrite,
                model=self.models.get(self.conf.matcher, None),
                matches_mode=self.conf.matches_mode,
                sparse_nms_radius=self.conf.sparse_nms_radius,
                verbose=self.conf.verbose,
            )
        if "depth" in self.match_dirs:
            self.depth_dir = self.match_dirs["depth"]
        self.models[self.conf.matcher] = model
        if "sparse" in self.conf.matches_mode:
            self.log(f"Matches located in {self.match_dirs['smatches']}", level=1)
        if "dense" in self.conf.matches_mode:
            self.log(f"Matches located in {self.match_dirs['dmatches']}", level=1)
        return matcher_conf


    def extract_pairwise(self, overwrite=False):
        """Extract all pairwise information."""
        print(f"Extracting pairwise {self.conf.matcher} information...")
        matcher_conf = load_cfg(PAIRWISE_CONFIG_DIR / f"{self.conf.matcher}.yaml")
        confs = {}
        if "sparse" in self.conf.matches_mode:
            features_conf = self.extract_sparse(overwrite=overwrite)
            confs["features"] = features_conf
        if matcher_conf.type == "sparse":
            matcher_conf = self.match_sparse(overwrite=overwrite)
        elif matcher_conf.type == "dense":
            matcher_conf = self.match_dense(overwrite=overwrite)
        else:
            raise NotImplementedError(f"Matcher {matcher_conf.type} not implemented")
        confs["matcher"] = matcher_conf
        return confs

    def extract_sparsefea_ag(self, overwrite=False):
        """Extract all pairwise information."""
        print(f"Extracting SuperPoint features {self.conf.matcher} information...")
        confs = {}
        if "sparse" in self.conf.matches_mode:
            features_conf = self.extract_sparse_ag(overwrite=overwrite)
            confs["features"] = features_conf
        return confs

    def extract_pairwise_ag(self, overwrite=False, i=None):
        """Extract all pairwise information."""
        print(f"Extracting pairwise {self.conf.matcher} information...")
        matcher_conf = load_cfg(PAIRWISE_CONFIG_DIR / f"{self.conf.matcher}.yaml")
        confs = {}
        if matcher_conf.type == "sparse":
            matcher_conf = self.match_sparse_ag(overwrite=overwrite, i=i)
        elif matcher_conf.type == "dense":
            matcher_conf = self.match_dense_ag(overwrite=overwrite, i=i)
        else:
            raise NotImplementedError(f"Matcher {matcher_conf.type} not implemented")
        confs["matcher"] = matcher_conf
        return confs

    def extract_mono(self, overwrite=False):
        """Extract monocular priors."""
        detph_conf = self.extract_depth(overwrite=overwrite)
        normals_conf = self.extract_normals(overwrite=overwrite)
        return detph_conf, normals_conf

    def extract_depth(self, overwrite=False):
        """Extract monocular depth."""
        print(f"Extracting {self.conf.depth} depth...")
        if any(s in self.extract for s in ["d", "depth"]):
            overwrite = True
        depth_conf = load_cfg(extract_mono.CONFIG_DIR / f"{self.conf.depth}.yaml")
        depth_conf.dataset = OmegaConf.merge(self.conf.dataset, depth_conf.dataset)

        self.depth_dir, model = extract_mono.main(
            depth_conf,
            self.cache_dir,
            scene_parser=self.scene_parser,
            overwrite=overwrite,
            image_list=self.images_list,
            model=self.models.get(depth_conf.name, None),
            verbose=self.conf.verbose,
        )
        self.models[depth_conf.name] = model
        self.log(f"Depth located in {self.depth_dir}", level=1)
        return depth_conf

    def extract_normals(self, overwrite=False):
        """Extract monocular normals."""
        if any(s in self.extract for s in ["n", "normals"]):
            overwrite = True
        print(f"Extracting {self.conf.normals} normals...")
        normals_conf = load_cfg(gvars.MONO_MODEL_CONFIG_DIR / f"{self.conf.normals}.yaml")
        normals_conf.dataset = OmegaConf.merge(self.conf.dataset, normals_conf.dataset)
        self.normals_dir, model = extract_mono.main(
            normals_conf,
            self.cache_dir,
            scene_parser=self.scene_parser,
            overwrite=overwrite,
            image_list=self.images_list,
            model=self.models.get(normals_conf.name, None),
            verbose=self.conf.verbose,
        )
        self.models[normals_conf.name] = model
        self.log(f"Normals located in {self.normals_dir}", level=1)
        return normals_conf

    def extract_retrieval(self, overwrite=False):
        """Extract retrieval features."""
        retrieval_conf = load_cfg(extract_features.CONFIG_DIR / f"{self.conf.retrieval}.yaml")
        print(f"Extracting {self.conf.retrieval} retrieval features...")
        if any(s in self.extract for s in ["r", "retrieval"]):
            overwrite = True

        self.retrieval_path, _ = extract_features.main(
            retrieval_conf,
            self.scene_parser,
            self.cache_dir,
            overwrite=overwrite,
            image_list=self.images_list,
            verbose=self.conf.verbose,
        )
        self.log(f"Retrieval features located in {self.retrieval_path}", level=1)
        return retrieval_conf

    def extract_retrieval_ag(self, overwrite=False):
        """Extract retrieval features."""
        retrieval_conf = load_cfg(extract_features.CONFIG_DIR / f"{self.conf.retrieval}.yaml")
        print(f"Extracting {self.conf.retrieval} retrieval features...")
        if any(s in self.extract for s in ["r", "retrieval"]):
            overwrite = True

        self.retrieval_path, _ = extract_features.main_ag(
            retrieval_conf,
            self.scene_parser,
            self.cache_dir,
            overwrite=overwrite,
            image_list=self.images_list,
            verbose=self.conf.verbose,
        )
        self.log(f"Retrieval features located in {self.retrieval_path}", level=1)
        return retrieval_conf

    def extract_pairs(self, pairs_type):
        """Extract pairs for matching."""
        print(f"Extracting {pairs_type} pairs...")
        if "example" in str(self.sfm_outputs_dir):  # release
            base = Path(self.cache_dir)
        else:
            base = Path(
                self.cache_dir,
                self.sfm_outputs_dir.parts[(-5) if "/zip/" in str(self.sfm_outputs_dir) else (-4)],
                *self.sfm_outputs_dir.parts[-2:],
            )
        base.mkdir(parents=True, exist_ok=True)
        if pairs_type == "exhaustive":
            self.sfm_pairs_path = Path(base, "pairs_exhaustive.txt")
            pairs_from_exhaustive(self.sfm_pairs_path, self.images_list)
        elif pairs_type == "sequential":
            self.sfm_pairs_path = Path(
                base, f"pairs_sequential-{self.conf.noverlap}-{self.conf.quadratic_overlap}.txt"
            )
            self.images_list = list(self.images_list)
            self.images_list = sorted(self.images_list)
            pairs_from_sequential(
                self.sfm_pairs_path,
                self.images_list,
                overlap=self.conf.noverlap,
                quadratic_overlap=self.conf.quadratic_overlap,
            )
        elif pairs_type == "retrieval":
            self.sfm_pairs_path = Path(base, f"pairs_{self.conf.retrieval}-{self.conf.nquery}.txt")
            pairs_from_retrieval(
                self.retrieval_path,
                self.sfm_pairs_path,
                self.conf.nquery,
                query_list=self.images_list,
                db_list=self.images_list,
            )
        elif pairs_type == "retrieval_ag":
            for i in range(len(self.scene_parser.imnames_unregister)):
                sfm_pairs_path = Path(base, f"pairs_one_{self.conf.retrieval}-{self.conf.nquery}-{i}.txt")
                self.sfm_pairs_paths.append(sfm_pairs_path)
                _, db_inpairs = pairs_from_retrieval(
                    self.retrieval_path,
                    sfm_pairs_path,
                    self.conf.nquery,
                    query_list=[self.scene_parser.imnames_unregister[i]],
                    db_list=self.scene_parser.imnames_register,
                    append=False,
                )

                nquery = min(len(db_inpairs), self.conf.nquery)
                pairs_from_retrieval(
                    self.retrieval_path,
                    sfm_pairs_path,
                    nquery,
                    query_list=db_inpairs,
                    db_list=db_inpairs,
                )
                image_inpair = [self.scene_parser.imnames_unregister[i]] + db_inpairs
                self.image_inpairs.append(image_inpair)
        elif pairs_type == "retrieval_ag_loca":
            self.imageKDIndex = build_image_kdtree(self.scene_parser.rec_origin)
            pairnames = []
            for i in range(len(self.scene_parser.imnames_unregister)):
                sfm_pairs_path = Path(base, f"pairs_one_{self.conf.retrieval}-{self.conf.nquery}-{i}.txt")
                self.sfm_pairs_paths.append(sfm_pairs_path)
                pairs, db_inpairs = pairs_from_retrieval(
                    self.retrieval_path,
                    sfm_pairs_path,
                    self.conf.nquery,
                    query_list=[self.scene_parser.imnames_unregister[i]],
                    db_list=self.scene_parser.imnames_register,
                    append=False,
                )
                pairnames.append(pairs)
                image_inpair = [self.scene_parser.imnames_unregister[i]] + db_inpairs
                self.image_inpairs.append(image_inpair)
            return pairnames

        return

    def extract_sky_mask(self, overwrite=False):
        """Extract sky masks."""
        if any(s in self.extract for s in ["s", "sky"]):
            overwrite = True
        skyseg_conf = load_cfg(extract_skyseg.CONFIG_DIR / "skyseg.yaml")
        skyseg_conf.dataset = OmegaConf.merge(self.conf.dataset, skyseg_conf.dataset)
        self.skyseg_dir, model = extract_skyseg.main(
            skyseg_conf,
            self.cache_dir,
            overwrite=overwrite,
            image_list=self.images_list,
            scene_parser=self.scene_parser,
            verbose=self.conf.verbose,
        )
        self.models[skyseg_conf.name] = model
        self.log(f"Sky segmentation located in {self.skyseg_dir}", level=1)

    def extract_masks(self, masks, overwrite=False):
        """Extract masks."""
        print(f"Extracting {masks} masks...")
        self.masks_dirs = []
        for mask in masks:
            if mask == "sky":
                self.extract_sky_mask(overwrite=overwrite)
                self.masks_dirs.append(self.skyseg_dir)
            else:
                raise NotImplementedError

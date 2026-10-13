from copy import deepcopy
from itertools import chain

import numpy as np
import pycolmap

from mpsfm.baseclass import BaseClass
from mpsfm.extraction.pairwise import CONFIG_DIR as PAIRWISE_CONFIG_DIR
from mpsfm.utils.io import get_matches
from mpsfm.utils.parsers import read_unique_pairs
from mpsfm.utils.tools import load_cfg

from .utils import (
    gather_dense_2view,
    gather_sparse_keypoints,
    gather_sparse_matches,
    geometric_verification,
)


class ColmapCorrespondenceGraphWrapper:
    def __getattr__(self, name):
        return getattr(self.cg, name)


class Correspondences(BaseClass, ColmapCorrespondenceGraphWrapper):
    """MP-SfM Correspondence object. Wraps COLMAP CorrespondenceGraph and handles everything related,
    including geometric verification."""

    default_conf = {
        "matches_mode": "<--->",
        "cached_dense_scores": False,
        "verbose": 0,
    }

    def _init(self, mpsfm_rec=None, extractor=None, sfm_outputs_dir=None, i=None):
        self.cg = pycolmap.CorrespondenceGraph()
        self.mpsfm_rec = mpsfm_rec
        self.extractor = extractor
        self._two_view_geom = {}
        self.sfm_outputs_dir = sfm_outputs_dir
        self.keypoints_set = None
        self.matches_set = None
        self.sparse_im_masks = None
        self.inlier_match_scores = None
        self.i = i

    # --- Geometry and Correspondence Access ---
    def two_view_geom(self, imname1, imname2):
        """two view geometry between image pair"""
        if (imname1, imname2) in self._two_view_geom:
            return self._two_view_geom[(imname1, imname2)], True
        if (imname2, imname1) in self._two_view_geom:
            two_view_geom = deepcopy(self._two_view_geom[(imname2, imname1)])
            two_view_geom.invert()
            return two_view_geom, True
        return None, False

    def matches(self, imid1, imid2):
        """matches between image pair"""
        return self.find_correspondences_between_images(imid1, imid2)

    # --- Data Import and Preparation ---
    def gather_correspondences(self, pairs, ims):
        """Gather correspondences from extractor"""
        matcher_conf = load_cfg(PAIRWISE_CONFIG_DIR / f"{self.extractor.conf.matcher}.yaml")
        if matcher_conf.type == "sparse":
            keypoints = gather_sparse_keypoints(self.extractor, ims)
            matches, scores = gather_sparse_matches(self.extractor, pairs)
            sparse_im_masks = None
        elif matcher_conf.type == "dense":
            keypoints, matches, scores, sparse_im_masks = gather_dense_2view(
                self.extractor,
                pairs,
                ims,
                matches_mode=self.conf.matches_mode,
            )
        else:
            raise ValueError(f"Unknown matcher type: {matcher_conf.type}")
        return keypoints, matches, scores, sparse_im_masks

    def gather_matches_scores(self, inlier_masks, scores, matches_set):
        """Gather match scores from inlier masks"""
        inlier_match_scores = {}
        for names, tvg in self._two_view_geom.items():
            if len(tvg.inlier_matches) == 0:
                inlier_match_scores[frozenset(names)] = 0
                continue
            if self.conf.cached_dense_scores:
                m = self.extractor.match_dirs["cache_matches"]
                _, s = get_matches(m, *names)
                if "dense" in self.conf.matches_mode and "sparse" in self.conf.matches_mode:
                    m = matches_set[names]
                    sparseA = self.sparse_im_masks[names[0]][m[:, 0]]
                    inlier_match_scores[frozenset(names)] = sum(s) if (~sparseA).sum() > 0 else 0
                else:
                    inlier_match_scores[frozenset(names)] = sum(s)
            else:
                inlm = inlier_masks[names]
                inlier_match_scores[frozenset(names)] = sum(scores[frozenset(names)][inlm])
        return inlier_match_scores

    def process_and_verify_matches(self):
        """Process correspondences and perform geometric verification"""
        if self.i is None:
            self.pairs = read_unique_pairs(self.extractor.sfm_pairs_path)
        else:
            self.pairs = read_unique_pairs(self.extractor.sfm_pairs_paths[self.i])
        self.ims = set(chain(*self.pairs))
        self.keypoints_set, self.matches_set, self.scores, self.sparse_im_masks = self.gather_correspondences(self.pairs, self.ims)
        inlier_masks, self._two_view_geom = geometric_verification(
            self.mpsfm_rec, self.pairs, keypoints=self.keypoints_set, matches=self.matches_set
        )
        self.inlier_match_scores = self.gather_matches_scores(inlier_masks, self.scores, self.matches_set)

    @staticmethod
    def read_last_pair(path):
        """Return the pair from the last line of the file (exactly last line)."""
        with open(path, "r") as f:
            lines = f.readlines()

        if not lines:
            raise ValueError(f"Empty pairs file: {path}")

        last_line = lines[-1].rstrip("\n")
        parts = last_line.split()

        if len(parts) < 2:
            raise ValueError(f"Last line does not contain a valid pair: {last_line!r}")

        pair = sorted(parts[:2])
        return [pair]  # ✅ List[List[str]]

    def process_and_verify_matches_add(self, rec_name_to_id):
        """Process correspondences and perform geometric verification"""
        if self.i is None:
            pair = self.read_last_pair(self.extractor.sfm_pairs_path)
        else:
            pair = self.read_last_pair(self.extractor.sfm_pairs_paths[self.i])
        ims = set(chain(*pair))
        self.pairs.append(pair[0])
        keypoints_set, matches_set, scores, sparse_im_masks = self.gather_correspondences(pair, ims)

        name0, name1 = pair[0]
        # add
        off0 = len(self.keypoints_set[name0]) if name0 in self.keypoints_set else 0
        off1 = len(self.keypoints_set[name1]) if name1 in self.keypoints_set else 0

        o1 = len(self.mpsfm_rec.images[rec_name_to_id[name0]].points2D)
        o2 = len(self.mpsfm_rec.images[rec_name_to_id[name1]].points2D)

        for n in (name0, name1):
            if n in self.keypoints_set:
                self.keypoints_set[n] = np.concatenate([self.keypoints_set[n], keypoints_set[n]], axis=0)
            else:
                self.keypoints_set[n] = keypoints_set[n]

        if sparse_im_masks is not None:
            for n in (name0, name1):
                if n in self.sparse_im_masks:
                    self.sparse_im_masks[n] = np.concatenate([self.sparse_im_masks[n], sparse_im_masks[n]], axis=0)
                else:
                    self.sparse_im_masks[n] = sparse_im_masks[n]

        m = matches_set[(name0, name1)].copy()
        m[:, 0] += off0
        m[:, 1] += off1
        self.matches_set[(name0, name1)] = m.astype(np.int32)

        print(f"After offset: max idx0={m[:, 0].max()}, max idx1={m[:, 1].max()}")
        print(f"kps0 total={len(self.keypoints_set[name0])}, kps1 total={len(self.keypoints_set[name1])}")

        assert m[:, 0].max() < len(self.keypoints_set[name0]), \
            f"idx0 out of bounds: {m[:, 0].max()} >= {len(self.keypoints_set[name0])}"
        assert m[:, 1].max() < len(self.keypoints_set[name1]), \
            f"idx1 out of bounds: {m[:, 1].max()} >= {len(self.keypoints_set[name1])}"

        for (n0, n1), matches in self.matches_set.items():
            kps0_len = len(self.keypoints_set[n0])
            kps1_len = len(self.keypoints_set[n1])
            if matches[:, 0].max() >= kps0_len or matches[:, 1].max() >= kps1_len:
                print(
                    f"MATCH OVERFLOW: ({n0},{n1}) max=({matches[:, 0].max()},{matches[:, 1].max()}) vs kps=({kps0_len},{kps1_len})")

        fs = frozenset((name0, name1))
        self.scores[fs] = scores[fs]

        inlier_masks, self._two_view_geom = geometric_verification(
            self.mpsfm_rec, self.pairs, keypoints=self.keypoints_set, matches=self.matches_set
        )
        self.inlier_match_scores = self.gather_matches_scores(inlier_masks, self.scores, self.matches_set)

        for imname, off in ((name0, off0), (name1, off1)):
            im_id = rec_name_to_id[imname]

            kps_add = self.keypoints_set[imname][off:]

            pts_add = [
                pycolmap.Point2D(xy=kp[:2])
                for kp in kps_add.astype(np.float16)
            ]

            self.mpsfm_rec.images[im_id].points2D.extend(pts_add)

            new_kps = self.keypoints_set[imname]
            self.mpsfm_rec.images[im_id].depth.uncertainty_update = self.mpsfm_rec.images[im_id].depth.uncertainty_at_kps(new_kps)
            self.mpsfm_rec.images[im_id].depth.kps = new_kps

            # pts = list(self.mpsfm_rec.images[im_id].points2D)
            # pts.extend(pts_add)
            # self.mpsfm_rec.images[im_id].points2D = pts

            aaa = 1


    # --- Graph Population ---
    def populate(self, **kwargs):
        """Initializes correspondence graph"""
        self.process_and_verify_matches()
        for im_id, image in self.mpsfm_rec.images.items():
            imname = image.name
            kps = self.keypoints_set[imname]
            self.mpsfm_rec.images[im_id].points2D = [pycolmap.Point2D(xy=kp0_i) for kp0_i in kps.astype(np.float16)]
            self.cg.add_image(im_id, kps.shape[0])

        for (name0, name1), tvg in self._two_view_geom.items():
            self.cg.add_correspondences(
                self.mpsfm_rec.imid(name0),
                self.mpsfm_rec.imid(name1),
                tvg.inlier_matches.astype(np.uint32),
            )
        for imid in self.mpsfm_rec.images:
            if self.cg.num_correspondences_for_image(imid) == 0:
                print(f"Image {imid} has no correspondences")

        self.cg.finalize()
        return True

    def populate_add(self, rec_name_to_id, **kwargs):
        """Initializes correspondence graph"""
        self.cg = pycolmap.CorrespondenceGraph()
        self.process_and_verify_matches_add(rec_name_to_id)
        for im_id, image in self.mpsfm_rec.images.items():
            imname = image.name
            kps = self.keypoints_set[imname]
            aaa = len(self.mpsfm_rec.images[im_id].points2D)
            if aaa != kps.shape[0]:
                print("MISMATCH", imname, "points2D", aaa, "kps", kps.shape[0])

            self.cg.add_image(im_id, kps.shape[0])

        for (name0, name1), tvg in self._two_view_geom.items():
            self.cg.add_correspondences(
                self.mpsfm_rec.imid(name0),
                self.mpsfm_rec.imid(name1),
                tvg.inlier_matches.astype(np.uint32),
            )
        for imid in self.mpsfm_rec.images:
            if self.cg.num_correspondences_for_image(imid) == 0:
                print(f"Image {imid} has no correspondences")

        self.cg.finalize()
        return True

    def process_and_verify_retrieval_matches(self, pairs_list):
        """Process correspondences and perform geometric verification"""
        top_pairs_list = []
        for pairs in pairs_list:
            ims = set(chain(*pairs))
            self.keypoints_set, self.matches_set, scores, self.sparse_im_masks = self.gather_correspondences(pairs, ims)
            inlier_masks, self._two_view_geom = geometric_verification(
                self.mpsfm_rec, pairs, keypoints=self.keypoints_set, matches=self.matches_set
            )
            self.inlier_match_scores = self.gather_matches_scores(inlier_masks, scores, self.matches_set)

            n = self.extractor.conf.nquery_loca

            top_items = sorted(self.inlier_match_scores.items(), key=lambda kv: kv[1], reverse=True)[:n]
            top_pairs = [tuple(k) for k, _ in top_items]

            top_pairs_list.append(top_pairs)

        return top_pairs_list

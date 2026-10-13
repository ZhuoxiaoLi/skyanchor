from collections import defaultdict
from pathlib import Path

import numpy as np

from .baseeval import BaseAggregator, BaseEval


def relative_pose_error(i_t_j, i_t_j_gt):
    m_est_n = np.eye(4)
    m_est_n[:3] = i_t_j.matrix()
    m_gt_n = np.eye(4)
    m_gt_n[:3] = i_t_j_gt.matrix()
    dr = m_est_n @ np.linalg.inv(m_gt_n)

    trace = np.trace(dr[:3, :3])
    cos = np.clip((trace - 1) / 2, -1, 1)
    dr = np.abs(np.arccos(cos)) / np.pi * 180

    t_est = m_est_n[:3, 3]
    t_est = t_est / np.linalg.norm(t_est)
    t_gt = m_gt_n[:3, 3]
    t_gt = t_gt / np.linalg.norm(t_gt)
    dt = np.clip(np.dot(t_est, t_gt), -1, 1)
    dt = np.arccos(dt) * 180 / np.pi
    dt = np.minimum(dt, 180 - dt)
    return dr, dt


def cal_error_auc(errors, thresholds):
    sort_idx = np.argsort(errors)
    errors = np.array(errors.copy())[sort_idx]
    recall = (np.arange(len(errors)) + 1) / len(errors)
    errors = np.r_[0.0, errors]
    recall = np.r_[0.0, recall]
    aucs = []
    for t in thresholds:
        last_index = np.searchsorted(errors, t)
        recall[:last_index]
        recall[last_index - 1]
        r = np.r_[recall[:last_index], recall[last_index - 1]]
        e = np.r_[errors[:last_index], t]
        aucs.append(np.round((np.trapz(r, x=e) / t), 4))
    return aucs


class EvalRelativePose(BaseEval):
    default_conf = {"thresholds": [1, 5, 20]}

    def _summarize(self):
        errs = [self.results["summary"][f"AUC-max@{auc}"] for auc in self.conf.thresholds]
        errs = " | ".join([f"{err*100:.2f}" for err in errs])
        return f"{errs}"

    def compute(self, only_registered=False):
        errors = self.relative_pose_errors(only_registered)
        errors_per_image = {imname: np.mean(list(e.values())) for imname, e in errors["max"].items()}
        errors_R_per_image = {imname: np.mean(list(e.values())) for imname, e in errors["R"].items()}
        errors_t_per_image = {imname: np.mean(list(e.values())) for imname, e in errors["t"].items()}

        errors_all = sum([list(e.values()) for e in errors["max"].values()], [])
        errors_R_all = sum([list(e.values()) for e in errors["R"].values()], [])
        errors_t_all = sum([list(e.values()) for e in errors["t"].values()], [])

        auc = cal_error_auc(errors_all, self.conf.thresholds)
        aucR = cal_error_auc(errors_R_all, self.conf.thresholds)
        auct = cal_error_auc(errors_t_all, self.conf.thresholds)
        self.results["full_results"] = errors

        self.results["results"]["errors_per_image"] = errors_per_image
        self.results["results"]["errors_R_per_image"] = errors_R_per_image
        self.results["results"]["errors_t_per_image"] = errors_t_per_image

        for i, th in enumerate(self.conf.thresholds):
            self.results["results"][f"AUC-max@{th}"] = auc[i]
            self.results["results"][f"AUC-R@{th}"] = aucR[i]
            self.results["results"][f"AUC-t@{th}"] = auct[i]

        for name, res in self.results["results"].items():
            if isinstance(res, dict):
                self.results["summary"][name] = {key: f"{val:.2f}" for key, val in res.items()}
            else:
                self.results["summary"][name] = res
        self.results["success"] = True

    def relative_pose_errors(self, only_registered=False):
        images = self.estim_rec.images if not only_registered else self.estim_rec.registered_images
        image_names = [image.name for image in images.values()]
        gt_name_to_id = {
            Path(image.name).name: imid
            for imid, image in self.gt_rec.images.items()
            if Path(image.name).name in image_names
        }

        errors = {"max": defaultdict(dict), "R": defaultdict(dict), "t": defaultdict(dict)}
        for image_i in images.values():
            for image_j in images.values():
                if image_i.name == image_j.name:
                    continue
                if not image_i.has_pose or not image_j.has_pose:
                    for key in errors:
                        errors[key][image_i.name][image_j.name] = 180
                else:
                    gt_image_i = self.gt_rec.images[gt_name_to_id[image_i.name]]
                    gt_image_j = self.gt_rec.images[gt_name_to_id[image_j.name]]

                    cj_t_ci = image_j.cam_from_world * image_i.cam_from_world.inverse()

                    cj_tgt_ci = gt_image_j.cam_from_world * gt_image_i.cam_from_world.inverse()
                    rel_R, rel_t = relative_pose_error(cj_t_ci, cj_tgt_ci)

                    err = eval("max")(rel_R, rel_t).item()
                    err_R = rel_R.item()
                    err_t = rel_t.item()
                    errors["max"][image_i.name][image_j.name] = err
                    errors["R"][image_i.name][image_j.name] = err_R
                    errors["t"][image_i.name][image_j.name] = err_t

        return errors


class AggregateRelativePose(BaseAggregator):
    eval_obj = EvalRelativePose
    default_conf = {"thresholds": [1, 5, 20]}

    def setup(
        self,
        path_template,
        exp_dir,
        scene_desc=None,
        group_desc=None,
        conf=None,
        aggr_desc=None,
        recdescs=None,
        **kwargs,
    ):
        return self._setup(path_template, exp_dir, scene_desc, group_desc, conf, aggr_desc, recdescs, **kwargs)

    def aggregate(self, recdescs=None):
        errors = {key.split("-")[0]: defaultdict(list) for key in self.aggregated_evals}

        for scene, testsets in recdescs.items():
            for testset_id, imids in testsets.items():
                if f"{scene}-{testset_id}" not in self.aggregated_evals:
                    for image_i in imids:
                        for image_j in imids:
                            if image_i == image_j:
                                continue
                            errors[scene]["max"].append(180)
                            errors[scene]["R"].append(180)
                            errors[scene]["t"].append(180)
                else:
                    eval_obj = self.aggregated_evals[f"{scene}-{testset_id}"]
                    for image_i in eval_obj.results["full_results"]["max"]:
                        for image_j in eval_obj.results["full_results"]["max"][image_i]:
                            errors[scene]["max"].append(eval_obj.results["full_results"]["max"][image_i][image_j])
                            errors[scene]["R"].append(eval_obj.results["full_results"]["R"][image_i][image_j])
                            errors[scene]["t"].append(eval_obj.results["full_results"]["t"][image_i][image_j])
        n_ims = len(sum([sum([imids for imids in testsets.values()], []) for testsets in recdescs.values()], []))
        n_reg_ims = sum(eval_obj.results["num_registered_images"] for eval_obj in self.aggregated_evals.values())
        mauc_list = [cal_error_auc(e["max"], self.conf.thresholds) for e in errors.values()]
        mauc = [np.mean([m[i] for m in mauc_list]) for i in range(len(self.conf.thresholds))]

        self.results = {
            "num_images": n_ims,
            "num_registered_images": n_reg_ims,
            "AUC-mmax": mauc,
        }

    def _summarize(self):
        s = [f"{el*100:.1f}" for el in self.results["AUC-mmax"]]
        summary = "/".join(s) + " " + f"({self.results['num_registered_images']}/{self.results['num_images']})"
        return summary

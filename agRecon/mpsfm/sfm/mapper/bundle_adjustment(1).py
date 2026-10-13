import numpy as np
import pyceres
import pycolmap
from pyceres import Problem
from pycolmap import LossFunctionType

from mpsfm.baseclass import BaseClass


def fit_robust_gaussian_mad(data):
    """Fits a robust Gaussian to the data using the Median Absolute Deviation (MAD) method."""
    mu = np.median(data)
    mad = np.median(np.abs(data - mu))
    sigma = 1.4826 * mad
    return mu, sigma


class Optimizer(BaseClass):
    """Optimizer class for Bundle Adjustment."""

    default_conf = {
        "depth_loss_name": "cauchy",
        "ref3d_loss_name": "trivial",
        "reproj_loss_name": "SOFT_L1",
        "reproj_loss_scale": 1.5,
        "scale_filter": True,
        "scale_filter_factor": 1.5,
        "metric_scale_filter": True,
        "rob_std": 2,
        "truncation_mode": "mad",  # [quantile, mad]
        # dev
        "gross_outliers": False,
        "single_rescale": True,
        "min_truncation_mult": None,
        "verbose": 0,
    }

    def _init(self, mpsfm_rec, correspondences):
        self.mpsfm_rec = mpsfm_rec
        self.correspondences = correspondences

        self.truncation_multiplier = 1

        self.get_loss = {
            "trivial": LossFunctionType.TRIVIAL,
            "cauchy": LossFunctionType.CAUCHY,
            "softl1": LossFunctionType.SOFT_L1,
        }

    def __yield_problem_parameters(self, optim_ids, proj_depths=False):
        for imid in optim_ids:
            image = self.mpsfm_rec.images[imid]
            camera = self.mpsfm_rec.rec.cameras[image.camera_id]
            pt2D_ids = image.get_observation_point2D_idxs()
            kps_with3D = image.keypoint_coordinates(pt2D_ids)
            p3d_ids = image.point3D_ids(pt2D_ids)

            kwargs = {"imid": imid, "image": image, "camera": camera, "pt3D_ids": p3d_ids, "kps": kps_with3D}
            kwargs["obsdepths"] = self.mpsfm_rec.images[imid].depth.data_prior_at_kps(kps_with3D)
            kwargs["valid"] = self.mpsfm_rec.images[imid].depth.valid_at_kps(kps_with3D)

            if proj_depths:
                _, _, _, depth3d, _ = self.mpsfm_rec.project_image_3d_points(imid, kwargs["pt3D_ids"])
                kwargs["projdepths"] = depth3d
            yield kwargs

    def __build_problem(
        self,
        bundle,
        fix_pose,
        fix_scale,
        mode=None,
        depth_loss_name=None,
        allow_scale_filter=False,
        param_multiplier=1,
        depth_type="update",
        **kw,
    ) -> tuple[Problem, dict, bool]:
        conf = self.conf
        optim_ids = list(bundle["optim_ids"])
        depth_loss_name = depth_loss_name or conf.depth_loss_name
        depth_loss_type = self.get_loss[depth_loss_name]
        shift_scale = {imid: np.array([0.0, 0.0]) for imid in optim_ids}

        ba_config = pycolmap.BundleAdjustmentConfig()
        for imid in optim_ids:
            ba_config.add_image(imid)
        if mode == "local":
            for p3Did in bundle["pts3D"]:
                if self.mpsfm_rec.points3D[p3Did].track.length() < 15:
                    ba_config.add_variable_point(p3Did)
        bundle_camids = [self.mpsfm_rec.images[c].camera_id for c in optim_ids]
        for camid in bundle_camids:
            ba_config.set_constant_cam_intrinsics(camid)
        # per image kp_std currently is not supported
        kp_std = np.median([self.mpsfm_rec.images[imid].kp_std for imid in optim_ids])

        options = pycolmap.BundleAdjustmentOptions(
            loss_function_magnitude=1 / kp_std**2,
            loss_function_type=pycolmap.LossFunctionType(self.conf.reproj_loss_name),
            loss_function_scale=self.conf.reproj_loss_scale * kp_std,
        )

        bundler = pycolmap.create_default_bundle_adjuster(options, ba_config, self.mpsfm_rec.rec)

        problem = bundler.problem
        scale_filter_factor = self.conf.scale_filter_factor
        gross_outliers = self.conf.gross_outliers
        param_multiplier *= self.truncation_multiplier

        for ii, imid in enumerate(optim_ids):
            image = self.mpsfm_rec.images[imid]
            pose = image.cam_from_world
            if fix_pose or ii == 0:
                problem.set_parameter_block_constant(pose.rotation.quat)
                problem.set_parameter_block_constant(pose.translation)
            else:
                if ii == 1:
                    problem.set_manifold(
                        pose.translation, pyceres.SubsetManifold(3, [0])
                    )  # fixing the scale of the problem
                problem.set_manifold(pose.rotation.quat, pyceres.EigenQuaternionManifold())

            if not image.depth.activated:
                continue

            image = self.mpsfm_rec.images[imid]
            p2Ds = np.array(image.get_observation_point2D_idxs())
            kps_with3D = image.keypoint_coordinates(p2Ds)
            valid = image.depth.valid_at_kps(kps_with3D)
            kps_with3D = kps_with3D[valid]
            if depth_type == "update":
                depths = image.depth.data_at_kps(kps_with3D)
            else:
                depths = image.depth.data_prior_at_kps(kps_with3D)
            p2Ds = p2Ds[valid]
            p3Ds = image.point3D_ids(p2Ds)
            _, _, _, depth3d, _ = self.mpsfm_rec.project_image_3d_points(imid, p3Ds)
            mask = depths > 0
            if allow_scale_filter and self.conf.scale_filter:
                div = depths / depth3d
                mask *= (div < scale_filter_factor) * (div > (1 / scale_filter_factor))
            uncertainty_update = image.depth.uncertainty_update
            variances = np.array([uncertainty_update[pt2D_id] for pt2D_id in p2Ds])
            if gross_outliers and image.depth.activated:
                whitened = np.abs(np.log(depths).clip(1e-6, None) - np.log(depth3d).clip(1e-6, None)) / variances**0.5
                mask *= whitened < 3

            if np.sum(mask) == 0:
                print("No valid points for depth regularizing")
                continue

            depths = depths[mask]
            p2Ds = p2Ds[mask]
            variances = variances[mask]
            inv_uncert = 1 / variances.clip(1e-6, None)
            p3Ds = np.array(p3Ds)[mask]

            m = param_multiplier * self.conf.rob_std
            params = m * variances**0.5 / depths
            magnitudes = depths**2 * inv_uncert

            pycolmap.create_depth_bundle_adjuster(
                problem,
                imid,
                p3Ds,
                depths,
                magnitudes,
                params,
                depth_loss_type,
                shift_scale[imid],
                self.mpsfm_rec.rec,
                fix_shift=True,
                fix_scale=fix_scale,
                logloss=True,
            )

            fix_shiftscale = [0]  # for now we do not support fixing optimizing shift
            if fix_scale:
                fix_shiftscale.append(1)
            if len(fix_shiftscale) > 0:
                problem.set_manifold(shift_scale[imid], pyceres.SubsetManifold(2, fix_shiftscale))

        self.solve(problem)
        return bundler, shift_scale

    def __build_problem_final(
        self,
        bundle,
        fix_pose,
        fix_scale,
        mode=None,
        depth_loss_name=None,
        allow_scale_filter=False,
        param_multiplier=1,
        depth_type="update",
        **kw,
    ) -> tuple[Problem, dict, bool]:
        conf = self.conf
        optim_ids = list(bundle["optim_ids"])
        depth_loss_name = depth_loss_name or conf.depth_loss_name
        depth_loss_type = self.get_loss[depth_loss_name]
        shift_scale = {imid: np.array([0.0, 0.0]) for imid in optim_ids}
        constant_ids = list(bundle["constant_ids"])

        ba_config = pycolmap.BundleAdjustmentConfig()
        for imid in optim_ids:
            ba_config.add_image(imid)
        if mode == "local":
            for p3Did in bundle["pts3D"]:
                if self.mpsfm_rec.points3D[p3Did].track.length() < 15:
                    ba_config.add_variable_point(p3Did)
        bundle_camids = [self.mpsfm_rec.images[c].camera_id for c in optim_ids]
        for camid in bundle_camids:
            ba_config.set_constant_cam_intrinsics(camid)
        # per image kp_std currently is not supported
        kp_std = np.median([self.mpsfm_rec.images[imid].kp_std for imid in optim_ids])

        options = pycolmap.BundleAdjustmentOptions(
            loss_function_magnitude=1 / kp_std**2,
            loss_function_type=pycolmap.LossFunctionType(self.conf.reproj_loss_name),
            loss_function_scale=self.conf.reproj_loss_scale * kp_std,
        )

        bundler = pycolmap.create_default_bundle_adjuster(options, ba_config, self.mpsfm_rec.rec)

        problem = bundler.problem
        scale_filter_factor = self.conf.scale_filter_factor
        gross_outliers = self.conf.gross_outliers
        param_multiplier *= self.truncation_multiplier

        for ii, imid in enumerate(optim_ids):
            image = self.mpsfm_rec.images[imid]
            pose = image.cam_from_world
            if imid in constant_ids:
                problem.set_parameter_block_constant(pose.rotation.quat)
                problem.set_parameter_block_constant(pose.translation)
            else:
                problem.set_manifold(pose.rotation.quat, pyceres.EigenQuaternionManifold())

            if not image.depth.activated:
                continue

            image = self.mpsfm_rec.images[imid]
            p2Ds = np.array(image.get_observation_point2D_idxs())
            kps_with3D = image.keypoint_coordinates(p2Ds)
            valid = image.depth.valid_at_kps(kps_with3D)
            kps_with3D = kps_with3D[valid]
            if depth_type == "update":
                depths = image.depth.data_at_kps(kps_with3D)
            else:
                depths = image.depth.data_prior_at_kps(kps_with3D)
            p2Ds = p2Ds[valid]
            p3Ds = image.point3D_ids(p2Ds)
            _, _, _, depth3d, _ = self.mpsfm_rec.project_image_3d_points(imid, p3Ds)
            mask = depths > 0
            if allow_scale_filter and self.conf.scale_filter:
                div = depths / depth3d
                mask *= (div < scale_filter_factor) * (div > (1 / scale_filter_factor))
            uncertainty_update = image.depth.uncertainty_update
            variances = np.array([uncertainty_update[pt2D_id] for pt2D_id in p2Ds])
            if gross_outliers and image.depth.activated:
                whitened = np.abs(np.log(depths).clip(1e-6, None) - np.log(depth3d).clip(1e-6, None)) / variances**0.5
                mask *= whitened < 3

            if np.sum(mask) == 0:
                print("No valid points for depth regularizing")
                continue

            depths = depths[mask]
            p2Ds = p2Ds[mask]
            variances = variances[mask]
            inv_uncert = 1 / variances.clip(1e-6, None)
            p3Ds = np.array(p3Ds)[mask]

            m = param_multiplier * self.conf.rob_std
            params = m * variances**0.5 / depths
            magnitudes = depths**2 * inv_uncert

            pycolmap.create_depth_bundle_adjuster(
                problem,
                imid,
                p3Ds,
                depths,
                magnitudes,
                params,
                depth_loss_type,
                shift_scale[imid],
                self.mpsfm_rec.rec,
                fix_shift=True,
                fix_scale=fix_scale,
                logloss=True,
            )

            fix_shiftscale = [0]  # for now we do not support fixing optimizing shift
            if fix_scale:
                fix_shiftscale.append(1)
            if len(fix_shiftscale) > 0:
                problem.set_manifold(shift_scale[imid], pyceres.SubsetManifold(2, fix_shiftscale))

        self.solve(problem)
        return bundler, shift_scale

    def __build_shiftscale_problem(
        self, bundle, allow_scale_filter=False, allow_metric_scale_filter=False
    ) -> tuple[Problem, dict, bool]:
        shift_scale = {}
        scale_filter = self.conf.scale_filter
        scale_filter_factor = self.conf.scale_filter_factor
        metric_scale_factor = self.conf.metric_scale_filter
        single_rescale = self.conf.single_rescale
        for kwargs in self.__yield_problem_parameters(
            bundle["optim_ids"], proj_depths=scale_filter or metric_scale_factor
        ):
            pose = kwargs["image"].cam_from_world
            imid, p3dids = kwargs["imid"], kwargs["pt3D_ids"]
            if (scale_filter_factor or metric_scale_factor) and (
                "ref_id" in bundle and imid != bundle["ref_id"] and single_rescale
            ):
                continue
            if (
                allow_metric_scale_filter
                and metric_scale_factor
                and ((imid == bundle["ref_id"]) or (not single_rescale))
            ):
                scale = kwargs["projdepths"] / (kwargs["obsdepths"].clip(1e-6, None))
                im_scale = self.mpsfm_rec.images[imid].depth.scale
                proposed_scale = scale * im_scale
                map_scale = np.mean(
                    [self.mpsfm_rec.images[id].depth.scale for id in bundle["optim_ids"] if id != imid]
                )
                div = map_scale / proposed_scale
                valid = (div < 1.5) * (div > (1 / 1.5))
                presum = kwargs["valid"].sum()

                kwargs["valid"] = kwargs["valid"] * valid
                if kwargs["valid"].sum() == 0:
                    print("WARNING: Settin all points as outliers for metric scale optim and using map scale!!")
                    shift_scale[imid] = np.array([0.0, np.log(map_scale / self.mpsfm_rec.images[imid].depth.scale)])
                    return shift_scale, True
                self.log(
                    f"Setting {presum - kwargs['valid'].sum()}"
                    f"points as outliers for metric scale optim, out of {presum}",
                    level=3,
                )
            if allow_scale_filter and scale_filter and not allow_metric_scale_filter:
                div = kwargs["obsdepths"] / kwargs["projdepths"]

                presum = kwargs["valid"].sum()
                kwargs["valid"] *= (div < scale_filter_factor) * (div > (1 / scale_filter_factor))
                print(f"Setting {presum - kwargs['valid'].sum()} points as outliers for scale optim")

            p3d = self.mpsfm_rec.point3D_coordinates(p3dids)
            z = (pose * p3d)[:, -1]
            z = z[kwargs["valid"]]
            odepth = kwargs["obsdepths"][kwargs["valid"]]
            proposed = np.median(np.log(((z / odepth)).clip(1e-6, None)))
            shift_scale[imid] = np.array([0.0, proposed])
        return shift_scale, True

    def calculate_point_covs(self, bundle):
        """Calculates point covariances for the given bundle."""
        ba_config = pycolmap.BundleAdjustmentConfig()
        for imid in bundle["optim_ids"]:
            ba_config.add_image(imid)
        for pt3D_id in bundle["pts3D"]:
            ba_config.add_variable_point(pt3D_id)
        bundle_camids = [self.mpsfm_rec.images[c].camera_id for c in bundle["optim_ids"]]
        for camid in bundle_camids:
            ba_config.set_constant_cam_intrinsics(camid)
        # per image kp_std currently is not supported
        kp_std = np.median([self.mpsfm_rec.images[imid].kp_std for imid in bundle["optim_ids"]])
        options = pycolmap.BundleAdjustmentOptions(loss_function_magnitude=1 / kp_std**2)
        bundler = pycolmap.create_default_bundle_adjuster(options, ba_config, self.mpsfm_rec.rec)
        options = pycolmap.BACovarianceOptions({"params": pycolmap.BACovarianceOptionsParams.POINTS})
        ba_cov = pycolmap.estimate_ba_covariance(options, self.mpsfm_rec.rec, bundler)
        if ba_cov is None:
            print("WARNING: No bundle adjustment applied")
            return False
        for p3Did in bundle["pts3D"]:
            self.mpsfm_rec.point_covs.data[p3Did] = ba_cov.get_point_cov(p3Did)

        return True

    def calculate_point_covs_final(self, bundle):
        """Calculates point covariances for the given bundle."""
        ba_config = pycolmap.BundleAdjustmentConfig()
        for imid in bundle["optim_ids"]:
            ba_config.add_image(imid)

        for pt3D_id in bundle["pts3D"]:
            ba_config.add_variable_point(pt3D_id)
        bundle_camids = [self.mpsfm_rec.images[c].camera_id for c in bundle["optim_ids"]]
        for camid in bundle_camids:
            ba_config.set_constant_cam_intrinsics(camid)
        # per image kp_std currently is not supported
        kp_std = np.median([self.mpsfm_rec.images[imid].kp_std for imid in bundle["optim_ids"]])
        options = pycolmap.BundleAdjustmentOptions(loss_function_magnitude=1 / kp_std**2)
        bundler = pycolmap.create_default_bundle_adjuster(options, ba_config, self.mpsfm_rec.rec)
        options = pycolmap.BACovarianceOptions({"params": pycolmap.BACovarianceOptionsParams.POINTS})
        ba_cov = pycolmap.estimate_ba_covariance(options, self.mpsfm_rec.rec, bundler)
        for p3Did in bundle["pts3D"]:
            self.mpsfm_rec.point_covs.data[p3Did] = ba_cov.get_point_cov(p3Did)

    def ba(self, bundle, mode, **kwargs) -> tuple[Problem, bool]:
        """Optimizes per frame data and 3d points in entire reconstruction"""
        if "constant_ids" in bundle:
            problem, _ = self.__build_problem_final(bundle, fix_pose=False, fix_scale=True, mode=mode, **kwargs)
        else:
            problem, _ = self.__build_problem(bundle, fix_pose=False, fix_scale=True, mode=mode, **kwargs)
        return problem, True

    def optimize_prior_shiftscale(self, bundle, **kwargs) -> tuple[dict, bool]:
        """Optimizes shift and scale of depth maps across all frames and sets them as activated"""
        shift_scale, success = self.__build_shiftscale_problem(bundle, **kwargs)
        if not success:
            return None, False
        shift_scale = {imid: (shift, np.exp(scale)) for imid, (shift, scale) in shift_scale.items()}
        return shift_scale, True

    def refine_3d_points(self, bundle, **kwargs) -> tuple[Problem, bool]:
        """Refines triangulated 3d points with depth maps keeping poses fixed. \
            If depth maps are not activated, only the reprojection erors minimized"""

        problem, _ = self.__build_problem(
            bundle, fix_pose=True, fix_scale=True, depth_loss_name=self.conf.ref3d_loss_name, **kwargs
        )
        return problem, True

    def solve(self, problem):
        """Solves the optimization problem."""
        options = pyceres.SolverOptions()
        options.linear_solver_type = pyceres.LinearSolverType.SPARSE_SCHUR
        options.minimizer_progress_to_stdout = bool(self.conf.verbose > 3)
        options.num_threads = -1
        summary = pyceres.SolverSummary()
        pyceres.solve(options, problem, summary)
        self.log(summary.BriefReport(), level=2)

    def update_truncation_multiplier(self, imids):
        """Updates the truncation multiplier based on the depth statistics of the given images.
        Proxy to help find outliers."""
        D3d, D, dstds, D3dunscaled = [], [], [], []
        Dunscaled = []
        for imid in imids:
            image = self.mpsfm_rec.images[imid]

            p2Ds = np.array(image.get_observation_point2D_idxs())
            kps_with3D = image.keypoint_coordinates(p2Ds)
            valid = image.depth.valid_at_kps(kps_with3D)
            kps_with3D = kps_with3D[valid]
            depths = image.depth.data_at_kps(kps_with3D)
            p2Ds = p2Ds[valid]
            p3Ds = np.array(image.point3D_ids(p2Ds))
            mask = depths > 0
            _, _, _, depth3d, _ = self.mpsfm_rec.project_image_3d_points(imid, p3Ds[mask])
            depths = depths[mask]
            uncertainty_update = image.depth.uncertainty_update
            variances = np.array([uncertainty_update[pt2D_id] for pt2D_id in p2Ds[mask]])

            D.append(depths)
            Dunscaled.append(depths / image.depth.scale)
            D3d.append(depth3d)
            D3dunscaled.append(depth3d / image.depth.scale)
            dstds.append(variances**0.5)

        if len(D3d) == 0 or len(D3dunscaled) == 0 or len(D3d) == 0 or len(D3dunscaled) == 0:
            return False

        depths = np.concatenate(D)
        depth3ds = np.concatenate(D3d)
        dstds = np.concatenate(dstds)

        log_stds = dstds / depths
        log_stds = np.clip(log_stds, 1e-6, None)
        log_distances = np.log(depths) - np.log(depth3ds)
        witened_log_distances = log_distances / log_stds
        _, sigma = fit_robust_gaussian_mad(witened_log_distances)
        self.truncation_multiplier = sigma
        if self.conf.min_truncation_mult is not None:
            self.truncation_multiplier = max(self.truncation_multiplier, self.conf.min_truncation_mult)

        return True

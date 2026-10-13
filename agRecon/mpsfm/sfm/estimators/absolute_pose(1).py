import pycolmap

from mpsfm.baseclass import BaseClass


class AbsolutePose(BaseClass):
    """Absolute pose estimation using COLMAP."""

    default_conf = {
        "colmap_estimation_options": pycolmap.AbsolutePoseEstimationOptions().todict(),
        "colmap_refinement_options": pycolmap.AbsolutePoseRefinementOptions().todict(),
        "verbose": 0,
    }
    default_conf["colmap_estimation_options"]["ransac"]["min_inlier_ratio"] = 0.1
    default_conf["colmap_estimation_options"]["ransac"]["max_error"] = 4.0
    default_conf["colmap_estimation_options"]["ransac"]["min_num_trials"] = 1000
    default_conf["colmap_estimation_options"]["ransac"]["max_num_trials"] = 50000

    # default_conf["colmap_refinement_options"]["max_num_iterations"] = 200
    # default_conf["colmap_refinement_options"]["gradient_tolerance"] = 1e-6
    # default_conf["colmap_refinement_options"]["loss_function_scale"] = 0.5
    # default_conf["colmap_refinement_options"]["refine_focal_length"] = False
    # default_conf["colmap_refinement_options"]["refine_extra_params"] = False

    def __call__(self, points2D, points3D, camera):
        estim_options = dict(self.conf.colmap_estimation_options)
        estim_options["ransac"] = pycolmap.RANSACOptions(**estim_options["ransac"])
        return pycolmap.estimate_and_refine_absolute_pose(
            points2D,
            points3D,
            camera,
            estimation_options=pycolmap.AbsolutePoseEstimationOptions(**estim_options),
            refinement_options=pycolmap.AbsolutePoseRefinementOptions(**self.conf.colmap_refinement_options),
        )

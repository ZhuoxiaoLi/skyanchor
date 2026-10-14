#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from argparse import ArgumentParser, Namespace
import sys
import os


class GroupParams:
    pass


class ParamGroup:
    def __init__(self, parser: ArgumentParser, name : str, fill_none = False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None
            if shorthand:
                if t == bool:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true")
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action="store_true")
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group


class ModelParams(ParamGroup):
    def __init__(self, parser, sentinel=False):
        self.sh_degree = 3
        self._source_path = ""
        self._model_path = ""
        self.init_ply_path = ""
        self.dense_init_ply_name = "points3D.ply"
        self.ground_dense_init_ply_name = ""
        self.init_scale_factor = 1.0
        self.init_opacity = 0.1
        self._images = "images"
        self._resolution = -1
        self._white_background = False
        self.data_device = "cpu"  # "cpu"
        # Image decoding workers; use 1 for sequential loading.
        self.camera_workers = 128
        self.eval = False
        self.ground_source_path = ""
        self.ground_images = "images"
        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        if getattr(g, 'ground_source_path', '') and g.ground_source_path.strip():
            g.ground_source_path = os.path.abspath(g.ground_source_path)
        return g


class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.separate_sh = True
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False
        super().__init__(parser, "Pipeline Parameters")


class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.iterations = 60000
        self.position_lr_init = 0.000016
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.lr_rate = 1.0  # Global LR multiplier
        self.position_lr_max_steps = 30000
        self.feature_lr = 0.0025
        self.shfeature_lr = 0.005
        self.opacity_lr = 0.025
        self.scaling_lr = 0.001
        self.rotation_lr = 0.001
        self.percent_dense = 0.001
        self.lambda_dssim = 0.2
        self.densification_interval = 600
        self.opacity_reset_interval = 9000
        self.opacity_reset_max = 0.01
        self.densify_from_iter = 2000
        self.densify_until_iter = 20000
        self.densify_grad_threshold = 0.00016
        self.densify_debug = False
        self.cull_opacity = 0.005
        self.random_background = False
        self.optimizer_type = "default"
        self.budget = 8000000
        self.split_distance = 0.15
        self.opacity_reduction = 0.95

        # Dense initialization
        self.use = True
        self.matches_per_ref= 10000 # number of matches per reference
        self.num_refs=2000 # number of reference images
        self.nns_per_ref= 3 # number of nearest neighbors per reference
        self.proj_err_tolerance= 0.005
        self.roma_model="outdoors" # you can change this to "indoors" or "outdoors"
        self.add_SfM_init = False
        self.pair_file=""
        self.ground_pair_file=""
        # Black mask pixels denote sky and are excluded from matching.
        self.roma_mask_dir=""

        self.filter_min_depth = 0.1
        self.filter_max_depth = 500
        self.filter_min_tri_angle = 1.0  # Degrees
        self.filter_sor_k = 30
        self.filter_sor_std = 2.0
        self.filter_use_sor = True

        self.roma_certainty_thresh = 0.5

        # Positive stage lengths override ground_start_iter and iterations.
        self.aerial_stage_iterations = 30000
        self.ground_stage_iterations = 30000
        self.ground_start_iter = 30000
        self.joint_finetune_iterations = 40000

        super().__init__(parser, "Optimization Parameters")


def get_combined_args(parser: ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k,v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)

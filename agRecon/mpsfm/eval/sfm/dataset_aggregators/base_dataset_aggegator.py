from omegaconf import OmegaConf


class BaseDatasetAggregator:
    aggregation_approach = None
    base_default_conf = {}
    default_conf = {}
    dataset_benchmark = None
    testset_type = None

    def __init__(self, conf=None, sfm_configs=None, scenes=None, **kwargs):
        if sfm_configs is None:
            sfm_configs = []
        if conf is None:
            conf = {}
        default_conf = OmegaConf.merge(
            OmegaConf.create(self.base_default_conf),
            OmegaConf.create(self.default_conf),
        )
        if isinstance(conf, dict):
            conf = OmegaConf.create(conf)
        self.conf = OmegaConf.merge(default_conf, conf)
        self.aggregators = {}

        if scenes is None:
            self.scenes = self.dataset.scenes
        else:
            self.scenes = scenes

        self.sfm_configs = sfm_configs
        self.benchmark_obj = self.dataset_benchmark({})
        self._init(**kwargs)

    def all_scenes(self, **kwargs):
        out = {}
        for sfm_conf in self.sfm_configs:
            if sfm_conf not in self.aggregators:
                self.aggregators[sfm_conf] = self.aggregation_approach({})
            success = self.aggregators[sfm_conf].setup(
                self.benchmark_obj.sfm_outputs_dir_template,
                self.dataset.default_exp_dir,
                "*",
                self.testset_type,
                sfm_conf,
                recdescs=self.recdescs,
            )
            if not success:
                continue
            self.aggregators[sfm_conf].aggregate(self.recdescs)
            out[sfm_conf] = self.aggregators[sfm_conf].summarize(**kwargs)
        return out

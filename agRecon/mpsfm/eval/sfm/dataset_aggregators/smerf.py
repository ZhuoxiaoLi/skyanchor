from mpsfm.data_proc.smerf import SMERFDataset
from mpsfm.eval.sfm.dataset_aggregators.base_dataset_aggegator import BaseDatasetAggregator
from mpsfm.eval.sfm.relative_pose import AggregateRelativePose
from mpsfm.test.smerf import SMERFTest


class SMERFAggregator(BaseDatasetAggregator):
    dataset = SMERFDataset
    dataset_benchmark = SMERFTest
    aggregation_approach = AggregateRelativePose
    default_conf = {}

    def _init(self, mode):
        self.recdescs = {}
        self.testset_type = f"{mode}"
        for scene in self.dataset.scenes:
            self.recdescs[scene] = self.benchmark_obj.read_testsets(scene, self.testset_type)["testsets"]

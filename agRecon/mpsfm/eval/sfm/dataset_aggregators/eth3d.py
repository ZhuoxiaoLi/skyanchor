from mpsfm.data_proc.eth3d import ETH3DDataset
from mpsfm.eval.sfm.dataset_aggregators.base_dataset_aggegator import BaseDatasetAggregator
from mpsfm.eval.sfm.relative_pose import AggregateRelativePose
from mpsfm.test.eth3d import ETH3DTest


class ETH3DAggregator(BaseDatasetAggregator):
    dataset = ETH3DDataset
    aggregation_approach = AggregateRelativePose
    dataset_benchmark = ETH3DTest
    default_conf = {}

    def _init(self, mode):
        self.recdescs = {}
        self.testset_type = f"{mode}"
        for scene in self.scenes:
            testset_dict = self.benchmark_obj.read_testsets(scene, self.testset_type)
            if testset_dict is None:
                print(f"{scene}-{mode} testset does not exist")
                continue
            self.recdescs[scene] = testset_dict["testsets"]

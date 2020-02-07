from metaworld.benchmarks import ML45WithPinnedGoal

from garage.experiment.meta_test_best_helper import MetaTestBestHelper

if __name__ == "__main__":
    MetaTestBestHelper.read_cmd(ML45WithPinnedGoal)

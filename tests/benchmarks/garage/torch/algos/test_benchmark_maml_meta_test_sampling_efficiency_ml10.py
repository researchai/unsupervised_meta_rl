from metaworld.benchmarks import ML10WithPinnedGoal

from garage.experiment.meta_test_best_helper import MetaTestBestHelper

if __name__ == "__main__":
    MetaTestBestHelper.read_cmd(ML10WithPinnedGoal)

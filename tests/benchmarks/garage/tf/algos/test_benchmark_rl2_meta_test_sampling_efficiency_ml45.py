from garage.experiment.meta_test_best_helper import MetaTestBestHelper

from metaworld.benchmarks import ML45

if __name__ == "__main__":
    MetaTestBestHelper.read_cmd(ML45.get_test_tasks)

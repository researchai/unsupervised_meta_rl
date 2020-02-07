from garage.experiment.meta_test_helper import MetaTestHelperTF

from metaworld.benchmarks import ML10

if __name__ == "__main__":
    MetaTestHelperTF.read_cmd(ML10.get_test_tasks, is_ml_45=False, is_normalized_reward=True)

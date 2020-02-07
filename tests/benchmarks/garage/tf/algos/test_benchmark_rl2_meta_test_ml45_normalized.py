from garage.experiment.meta_test_helper import MetaTestHelperTF

from metaworld.benchmarks import ML45

if __name__ == '__main__':
    MetaTestHelperTF.read_cmd(ML45.get_test_tasks,
                              is_ml_45=True,
                              is_normalized_reward=True)

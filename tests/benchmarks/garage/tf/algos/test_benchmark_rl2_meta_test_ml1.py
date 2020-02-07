from garage.experiment.meta_test_helper import MetaTestHelperTF

from metaworld.benchmarks import ML1

if __name__ == '__main__':
    MetaTestHelperTF.read_cmd(ML1.get_train_tasks('reach-v1'),
                              is_ml_45=False,
                              is_normalized_reward=False)

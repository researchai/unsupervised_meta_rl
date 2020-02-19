from garage.envs.ml_wrapper import ML10WithPinnedGoal

from garage.experiment.meta_test_helper import MetaTestHelper

if __name__ == "__main__":
    MetaTestHelper.read_cmd(ML10WithPinnedGoal.get_test_tasks)

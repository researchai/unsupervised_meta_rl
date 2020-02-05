from functools import partial
import os
import sys

from dowel import logger, StdOutput, CsvOutput, tabular, TensorBoardOutput

from metaworld.benchmarks import ML45WithPinnedGoal

from garage.envs.TaskIdWrapper import TaskIdWrapper
from garage.experiment import MetaEvaluator, Snapshotter
from garage.experiment import LocalRunner, SnapshotConfig
from garage.experiment.task_sampler import AllSetTaskSampler


meta_train_dirs = []
max_path_length = 150
adapt_rollout_per_task = 10
test_rollout_per_task = 10

meta_task_cls = ML45WithPinnedGoal.get_test_tasks


def resume_training(meta_train_dir):
    logger.add_output(CsvOutput(os.path.join(meta_train_dir, 'progress-resume.csv')))
    logger.add_output(TensorBoardOutput(meta_train_dir))
    logger.add_output(StdOutput())

    snapshot_config = SnapshotConfig(snapshot_dir=meta_train_dir,
                                     snapshot_mode='all',
                                     snapshot_gap=1)

    runner = LocalRunner(snapshot_config=snapshot_config)
    runner.restore(meta_train_dir)
    runner.resume()

    logger.remove_output_type(CsvOutput)
    logger.remove_output_type(StdOutput)
    logger.remove_output_type(TensorBoardOutput)

if __name__ == '__main__':
    assert len(sys.argv) == 2
    meta_train_dir = sys.argv[1]

    resume_training(meta_train_dir)

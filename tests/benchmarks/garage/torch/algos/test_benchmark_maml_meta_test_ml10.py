from functools import partial
import os
import sys

from dowel import logger, StdOutput, CsvOutput, tabular

from metaworld.benchmarks import ML10WithPinnedGoal

from garage.experiment import MetaEvaluator, Snapshotter
from garage.experiment import LocalRunner, SnapshotConfig
from garage.experiment.task_sampler import SetTaskSampler


meta_train_dirs = []
max_path_length = 150
rollout_per_task = 10
meta_batch_size = 20

meta_task_cls = ML10WithPinnedGoal.get_test_tasks


def do_meta_test(meta_train_dir, meta_test_dir):
    log_filename = os.path.join(meta_test_dir, 'meta-test.csv')
    logger.add_output(CsvOutput(log_filename))
    logger.add_output(StdOutput())

    snapshot_config = SnapshotConfig(snapshot_dir=meta_test_dir,
                                     snapshot_mode='all',
                                     snapshot_gap=1)

    runner = LocalRunner(snapshot_config=snapshot_config)
    meta_sampler = SetTaskSampler(meta_task_cls)
    runner.restore(meta_train_dir)

    meta_evaluator = MetaEvaluator(runner,
                                   test_task_sampler=meta_sampler,
                                   max_path_length=max_path_length,
                                   n_test_tasks=meta_batch_size,
                                   n_exploration_traj=rollout_per_task)

    itrs = Snapshotter.get_available_itrs(meta_train_dir)

    for itr in itrs:
        runner.restore(meta_train_dir, from_epoch=itr)
        meta_evaluator.evaluate(runner._algo)

        tabular.record('MetaTrain/Epoch', runner._stats.total_epoch)
        tabular.record('MetaTrain/TotalEnvSteps',
                       runner._stats.total_env_steps)
        logger.log(tabular)
        logger.dump_output_type(CsvOutput)

    logger.remove_output_type(CsvOutput)
    logger.remove_output_type(StdOutput)


if __name__ == '__main__':
    if len(sys.argv) > 1:
        meta_train_dirs = sys.argv[1:]

    for meta_train_dir in meta_train_dirs:
        meta_test_dir = meta_train_dir
        do_meta_test(meta_train_dir, meta_test_dir)

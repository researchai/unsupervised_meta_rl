"""MetaTestHelper for sampling efficiency."""
import argparse
import math
import os
import re

from dowel import logger, StdOutput, CsvOutput, tabular
import pandas as pd
import tensorflow as tf

from garage.experiment import MetaEvaluator
from garage.experiment import LocalRunner, SnapshotConfig
from garage.experiment.task_sampler import AllSetTaskSampler
from garage.tf.experiment import LocalTFRunner

adapt_rollouts_to_test = [0, 1, 2, 4, 8, 16, 32, 64]

class MetaTestBestHelper:
    def __init__(self,
                 meta_task_cls,
                 max_path_length=150,
                 adapt_rollout_per_task=10,
                 test_rollout_per_task=10):

        self.meta_task_cls = meta_task_cls
        self.max_path_length = max_path_length
        self.adapt_rollout_per_task = adapt_rollout_per_task
        self.test_rollout_per_task = test_rollout_per_task

        # random_init should be False in testing.
        self._set_random_init(False)

    @classmethod
    def read_cmd(cls, env_cls):
        logger.add_output(StdOutput())

        parser = argparse.ArgumentParser()
        parser.add_argument("folder", nargs="+")
        # Adaptation parameters
        parser.add_argument("--adapt-rollouts", nargs="?", default=10, type=int)
        parser.add_argument("--test-rollouts", nargs="?", default=20, type=int)
        parser.add_argument("--max-path-length", nargs="?", default=100, type=int)
        # Number of workers
        parser.add_argument("--parallel", action='store_true', default=False)
        # If itr==None,
        # pick the iteration with the best average training success rate.
        parser.add_argument("--itrs", nargs="*", type=int, default=None)

        args = parser.parse_args()
        meta_train_dirs = args.folder
        parallel = args.parallel
        adapt_rollout_per_task = args.adapt_rollouts
        test_rollout_per_task = args.test_rollouts
        max_path_length = args.max_path_length
        itrs = args.itrs

        helper = cls(
            meta_task_cls=env_cls,
            max_path_length=max_path_length,
            adapt_rollout_per_task=adapt_rollout_per_task,
            test_rollout_per_task=test_rollout_per_task)

        helper.test_many_folders(
            folders=meta_train_dirs,
            parallel=parallel,
            itrs=itrs)

    @classmethod
    def _get_tested_itrs(cls, meta_train_dir):
        files = [f for f in os.listdir(meta_train_dir) if f.endswith('.csv')]
        if not files:
            return []

        itrs = []
        for file in files:
            nums = re.findall(r'\d+', file)
            if nums:
                itrs.append(int(nums[0]))
        itrs.sort()

        return itrs

    @classmethod
    def _get_best_itr(cls, folder):
        df = pd.read_csv(os.path.join(folder, "progress.csv"))
        itr = df.iloc[df['Evaluation/SuccessRate'].idxmax()]['Evaluation/Iteration']
        sr = df['Evaluation/SuccessRate'].max()
        return int(itr), sr


    @classmethod
    def _set_random_init(cls, random_init):
        """Override metaworld's random_init"""
        from metaworld.envs.mujoco.env_dict import EASY_MODE_ARGS_KWARGS
        from metaworld.envs.mujoco.env_dict import MEDIUM_MODE_ARGS_KWARGS
        from metaworld.envs.mujoco.env_dict import HARD_MODE_ARGS_KWARGS

        def _set_random_init_kwargs(d):
            for _, task_kwargs in d.items():
                task_kwargs['kwargs']['random_init'] = random_init

        _set_random_init_kwargs(EASY_MODE_ARGS_KWARGS)
        _set_random_init_kwargs(MEDIUM_MODE_ARGS_KWARGS['train'])
        _set_random_init_kwargs(MEDIUM_MODE_ARGS_KWARGS['test'])
        _set_random_init_kwargs(HARD_MODE_ARGS_KWARGS['train'])
        _set_random_init_kwargs(HARD_MODE_ARGS_KWARGS['test'])

    def test_one_folder(self, meta_train_dir, itr=None):
        itr, sr = itr or self._get_best_itr(meta_train_dir)
        logger.log(
            'Load from {}th iteration of success rate {}'.format(itr, sr))

        snapshot_config = SnapshotConfig(snapshot_dir=meta_train_dir,
                                         snapshot_mode='all',
                                         snapshot_gap=1)

        with LocalTFRunner(snapshot_config=snapshot_config) as runner:
            meta_sampler = AllSetTaskSampler(self.meta_task_cls)
            runner.restore(meta_train_dir, from_epoch=itr)

            meta_evaluator = MetaEvaluator(
                runner,
                test_task_sampler=meta_sampler,
                max_path_length=self.max_path_length,
                n_test_tasks=meta_sampler.n_tasks,
                prefix='')

            log_filename = os.path.join(
                meta_train_dir, 'sampling-efficiency-best.csv')
            logger.log("Writing into {}".format(log_filename))
            logger.add_output(CsvOutput(log_filename))

            for adapt_rollouts in adapt_rollouts_to_test:
                logger.log("Testing {} adaptation rollouts".format(adapt_rollouts))

                meta_evaluator.evaluate(
                    runner._algo,
                    self.test_rollout_per_task,
                    adapt_rollouts
                )

                tabular.record('Iteration', runner._stats.total_epoch)
                tabular.record('TotalEnvSteps', runner._stats.total_env_steps)
                tabular.record('AdaptRollouts', adapt_rollouts)
                logger.log(tabular)

            logger.dump_output_type(CsvOutput)
            logger.remove_output_type(CsvOutput)

    def test_many_folders(self, folders, parallel, itrs=None):
        children = []
        for i, meta_train_dir in enumerate(folders):
            itr = itrs[i] if itrs else None
            if parallel:
                pid = os.fork()
                if pid == 0:
                    # In child process
                    self.test_one_folder(meta_train_dir, itr)
                    exit(0)
                else:
                    # In parent process
                    children.append(pid)
            else:
                self.test_one_folder(meta_train_dir, itr)

        if parallel:
            for child in children:
                os.waitpid(child, 0)

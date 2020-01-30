"""Test garage.experiment.LocalRunner"""
import tempfile
import unittest

import akro
import dowel
from dowel import logger
import numpy as np
import pytest

from garage import log_performance, TrajectoryBatch
from garage.envs import EnvSpec
from garage.experiment import SnapshotConfig
from garage.experiment.local_runner import LocalRunner


@pytest.mark.serial
def test_error_on_new_tabular_keys():
    lengths = np.array([2, 2])
    batch = TrajectoryBatch(
        EnvSpec(akro.Box(np.array([0., 0., 0.]), np.array([1., 1., 1.])),
                akro.Box(np.array([-1., -1.]), np.array([0., 0.]))),
        observations=np.ones((sum(lengths), 3), dtype=np.float32),
        last_observations=np.ones((len(lengths), 3), dtype=np.float32),
        actions=np.zeros((sum(lengths), 2), dtype=np.float32),
        rewards=np.zeros(sum(lengths), dtype=np.float32),
        terminals=np.array([0, 0, 0, 0], dtype=bool),
        env_infos={'success': np.array([0, 1, 0, 0], dtype=bool)},
        agent_infos={},
        lengths=lengths)

    logger.remove_all()
    log_file = tempfile.NamedTemporaryFile()
    csv_output = dowel.CsvOutput(log_file.name)
    logger.add_output(dowel.StdOutput())
    logger.add_output(csv_output)
    with tempfile.TemporaryDirectory() as log_dir_name:
        runner = LocalRunner(
            SnapshotConfig(snapshot_dir=log_dir_name,
                           snapshot_mode='last',
                           snapshot_gap=1))
        runner._start_worker = unittest.mock.MagicMock()
        runner.save = unittest.mock.MagicMock()
        runner._train_args = unittest.mock.MagicMock()
        runner._train_args.n_epochs = 10
        epochs = LocalRunner.step_epochs(runner)
        next(epochs)
        batch0 = batch._replace(env_infos={})
        log_performance(0, batch0, 0.8, prefix='test_log_performance')
        next(epochs)
        log_performance(1, batch, 0.8, prefix='test_log_performance')
        with pytest.raises(dowel.csv_output.CsvOutputWarning,
                           match='Inconsistent TabularInput keys'):
            next(epochs)
    logger.remove_all()

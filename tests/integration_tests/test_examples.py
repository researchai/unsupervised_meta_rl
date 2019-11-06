"""
This is an integration test to make sure scripts from examples/
work when executing `./examples/[*/]*.py`.
"""
import os
import subprocess
import tempfile
from unittest.mock import Mock

import pytest

from garage.envs import GridWorldEnv
from garage.np.policies import ScriptedPolicy
from garage.tf.envs import TfEnv

examples_root_dir = 'examples/'
blacklist = [
    os.path.join(examples_root_dir, 'resume_training.py'),
]
sim_policy_path = os.path.join(examples_root_dir, 'sim_policy.py')


def enumerate_examples():
    examples = []
    for (dirpath, _, filenames) in os.walk(examples_root_dir):
        for file in filenames:
            filepath = os.path.join(dirpath, file)
            if filepath not in blacklist and file.endswith(
                    '.py') and file != sim_policy_path:
                examples.append(filepath)
    return examples


@pytest.mark.parametrize('filepath', enumerate_examples())
def test_examples(filepath):
    args = [
        filepath,
    ]
    if filepath in {
            os.path.join(examples_root_dir, 'step_env.py'),
            os.path.join(examples_root_dir, 'step_dm_control_env.py')
    }:
        args.extend(['--n_max_steps', '1'])
    env = os.environ.copy()
    env['GARAGE_EXAMPLE_TEST_N_EPOCHS'] = '1'
    assert subprocess.run(args,
                          stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE,
                          check=True,
                          env=env).returncode == 0


class TestSimPolicy:

    def setup_method(self):
        env = TfEnv(GridWorldEnv(desc='4x4'))
        policy = ScriptedPolicy(
            scripted_actions=[2, 2, 1, 0, 3, 1, 1, 1, 2, 2, 1, 1, 1, 2, 2, 1])
        algo = Mock(env_spec=env.spec, policy=policy, max_path_length=16)
        self.fd = tempfile.NamedTemporaryFile()
        params = dict()
        # # Save arguments
        # params['setup_args'] = self._setup_args
        # params['train_args'] = self.train_args
        #
        # # Save states
        # params['env'] = self.env
        # params['algo'] = self.algo
        # params['paths'] = paths
        # params['last_epoch'] = epoch
        # pickle.dump(params, self.fd)

    @pytest.mark.skip(reason='incomplete')
    def test_sim_policy(self):
        args = [
            sim_policy_path, '--speedup', '1.5', '--max_path_length', '500',
            self.fd.name
        ]
        assert subprocess.run(args,
                              stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE,
                              check=True).returncode == 0

    def teardown_method(self):
        self.fd.close()

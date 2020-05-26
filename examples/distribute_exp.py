#!/usr/bin/env python3
from garage.experiment import distribute

config = {
    'gitanshu@172.18.1.3': 'examples/tf/trpo_cartpole.py',
    # 'resl@172.18.1.2': 'examples/tf/rl2_ppo_ml1.py',
    # 'resl@172.18.1.5': 'examples/tf/dqn_pong.py'
}

if __name__ == '__main__':
    distribute(config=config,
               exp_name_suffix='15',
               connect_kwargs={'password': 'reslresl'})

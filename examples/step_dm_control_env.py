#!/usr/bin/env python3
"""Example of how to load, step, and visualize an environment.

This example requires that garage[dm_control] be installed.
"""
import argparse

from garage.envs.dm_control import DmControlEnv

parser = argparse.ArgumentParser()
parser.add_argument('--n_max_steps',
                    type=int,
                    default=None,
                    help='Number of steps to run')
args = parser.parse_args()

# Construct the environment
env = DmControlEnv.from_suite('walker', 'run')

# Reset the environment and launch the viewer
env.reset()
env.render()

# Step randomly until interrupted
try:
    print('Press Ctrl-C to stop...')
    steps = 0
    while True:
        if args.n_max_steps:
            if steps == args.n_max_steps:
                break
            steps += 1
        env.step(env.action_space.sample())
        env.render()
except KeyboardInterrupt:
    print('Exiting...')
    env.close()

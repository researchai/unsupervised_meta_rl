#!/usr/bin/env python3
"""Example of how to load, step, and visualize an environment."""
import argparse

import gym

parser = argparse.ArgumentParser()
parser.add_argument('--n_max_steps',
                    type=int,
                    default=None,
                    help='Number of steps to run. If this argument is omitted')
args = parser.parse_args()
# Construct the environment

env = gym.make('MountainCar-v0')

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

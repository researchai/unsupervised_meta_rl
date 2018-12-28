import gym
import time

env = gym.make('CartPole-v1')
env.reset()

done = False

while True:
    time.sleep(1 / 60)
    env.render()
    if done:
        env.reset()
    _, _, done, _ = env.step(env.action_space.sample())


import gym
import policy_gradient
import torch
import torch.nn as nn

env = gym.make('CartPole-v0')
agent = policy_gradient.PolicyGradient(4, 32, 2, 0.01, nn.Tanh, nn.Softmax)
agent.train(env, render=False)
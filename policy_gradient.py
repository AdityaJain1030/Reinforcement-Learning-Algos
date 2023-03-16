import gym
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.optim import Adam
import numpy as np


class PolicyGradient:
    def __init__(self, input_size, hidden_layer_size, output_size, lr, activation, final_activation):
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_layer_size),
            activation(),
            nn.Linear(hidden_layer_size, hidden_layer_size),
            activation(),
            nn.Linear(hidden_layer_size, output_size),
            final_activation()
        )
        self.optimizer = Adam(self.model.parameters(), lr=lr)

    def get_action_distribution(self, obs):
        logits = self.model(obs)
        return Categorical(logits=logits)
    
    def sample_action(self, obs):
        return self.get_action_distribution(obs).sample().item()
    
    #takes mean because loss over the whole batch
    def loss(self, obs, action_taken, reward):
        logp = self.get_action_distribution(obs).log_prob(action_taken)
        return -(logp * reward).mean()
    
    def calculate_rewards(self, ep_rewards):
        rtgs = []
        for i in range(len(ep_rewards)):
            rtgs.append(sum(ep_rewards[i:]))
        return rtgs
    
    def train(self, env, batch_size = 5000, epochs = 50, render = False):
        for epoch in range(epochs):
            batch_obs = []
            batch_acts = []
            batch_rewards = []
            episode_rewards = []

            #metrics
            ep_len = []
            ep_ret = []

            obs = env.reset()
            ep_done = False
            while len(batch_obs) < batch_size:
                if render:
                    env.render()

                batch_obs.append(obs.copy())
                action = self.sample_action(torch.as_tensor(obs, dtype=torch.float32))
                batch_acts.append(action)
                obs, reward, ep_done, _ = env.step(action)
                episode_rewards.append(reward)

                if ep_done:
                    batch_rewards += self.calculate_rewards(episode_rewards)
                    ep_len.append(len(episode_rewards))
                    ep_ret.append(sum(episode_rewards))

                    #reset env 
                    obs = env.reset()
                    episode_rewards = []
                    ep_done = False
            
            batch_rewards += [sum(episode_rewards[i:]) for i in range(len(episode_rewards))]


            #update model
            self.optimizer.zero_grad()
            avg_loss = self.loss(
                torch.as_tensor(batch_obs, dtype=torch.float32),
                torch.as_tensor(batch_acts, dtype=torch.int64),
                torch.as_tensor(batch_rewards, dtype=torch.float32)
            )
            avg_loss.backward()
            self.optimizer.step()

            #metrics
            print(f"Epoch: {epoch}, Avg Loss: {avg_loss}, Avg Return: {np.mean(ep_ret)}, Avg Length: {np.mean(ep_len)}")

    def test(self, env):
        obs = env.reset()
        ep_done = False
        while True:
            env.render()
            action = self.sample_action(torch.as_tensor(obs, dtype=torch.float32))
            obs, _, ep_done, _ = env.step(action)
            if ep_done:
                obs = env.reset()
                ep_done = False
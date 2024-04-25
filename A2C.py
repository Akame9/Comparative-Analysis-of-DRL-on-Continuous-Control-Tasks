import os
import random
import cv2
import imageio
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import trange
import torch.distributions as distributions


# Define the model
class ActorCritic(nn.Module):
    def __init__(self, n_states, n_actions, hidden_size=512):
        super(ActorCritic, self).__init__()
        self.l1 = nn.Linear(n_states, hidden_size)
        self.l2 = nn.Linear(hidden_size, 256)
        self.action_mean = nn.Linear(256, n_actions)
        self.action_std = nn.Parameter(torch.zeros(n_actions))
        self.value_out = nn.Linear(256, 1)

    def forward(self, x):
        x = torch.relu(self.l1(x))
        x = torch.relu(self.l2(x))
        action_mean = self.action_mean(x)
        action_std = torch.exp(self.action_std).expand_as(action_mean)
        state_values = self.value_out(x)
        return action_mean, action_std, state_values

class A2CAgent:
    def __init__(self, env, lr=1e-4, gamma=0.99, seed=42, save_interval=10, save_dir='results'):
        self.seed = seed
        self.set_seed()
        self.env = env
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_states = int(sum(np.prod(spec.shape) for spec in env.observation_spec().values())) #sum(np.prod(spec["shape"]) for spec in env.observation_spec().values())   for soccer env
        self.n_actions = env.action_spec().shape[0] # 2 for soccer env
        self.model = ActorCritic(self.n_states, self.n_actions).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.gamma = gamma
        self.save_interval = save_interval
        self.save_dir = save_dir
        self.episode_rewards = []
        self.episode_losses = []

    def set_seed(self):
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

    def train(self, num_episodes=1000):

        for episode in trange(1,num_episodes+1):
            timestep = self.env.reset()
            episode_reward = 0
            saved_actions = []
            saved_rewards = []
            count =0
            while not timestep.last(): 
                count+=1 
                state = np.concatenate([np.asarray([ob]) if isinstance(ob, float) else ob for ob in timestep.observation.values()], axis=0)
                state = torch.tensor(state).float().to(self.device)
                
                action_mean, action_std, value = self.model(state)
                distribution  = distributions.Normal(action_mean, action_std)
                action = distribution.sample()
                log_prob = distribution.log_prob(action).sum()
                
                # Perform action
                action =torch.clamp(action, min=-1, max=-1) #-1,1
                env_action = action.clone().detach().cpu().numpy()
                timestep = self.env.step(env_action)
                episode_reward += timestep.reward
                
                # Save rewards and actions
                saved_rewards.append(timestep.reward)
                saved_actions.append((action, value, log_prob))

            loss = self.optimize_model(saved_rewards, saved_actions)
            self.episode_losses.append(loss)
            self.episode_rewards.append(episode_reward)

            print(f'Episode {episode}, Total Reward: {episode_reward}, Loss: {loss}, Steps: {count}')
            if episode % self.save_interval == 0:
                self.save_model(os.path.join(self.save_dir, f"a2c_walker_model_{episode}.pth"))

    def optimize_model(self, saved_rewards, saved_actions):
        R = 0
        saved_returns = []
        for r in saved_rewards[::-1]:
            R = r + self.gamma * R
            saved_returns.insert(0, R)

        saved_returns = torch.tensor(saved_returns)
        saved_returns = (saved_returns - saved_returns.mean()) / (saved_returns.std() + np.finfo(np.float32).eps.item())

        policy_losses = []
        value_losses = []
        for (action, value, log_prob), R in zip(saved_actions, saved_returns):
            advantage = R - value.item()
            policy_losses.append(-log_prob * advantage)
            value_losses.append(nn.functional.smooth_l1_loss(value, torch.tensor([R]).to(self.device)))
        
        self.optimizer.zero_grad()
        l1 = torch.stack(policy_losses).sum()
        l2 = torch.stack(value_losses).sum()
        loss = l1 + l2
        loss.backward()
        self.optimizer.step()

        return loss.item()
    
    
    def save_model(self, filepath):
        checkpoint = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'episode_rewards': self.episode_rewards,
            'episode_losses': self.episode_losses
        }
        torch.save(checkpoint, filepath)

    def load_model(self, filepath):
        checkpoint = torch.load(filepath)
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.episode_rewards = checkpoint['episode_rewards']
        self.episode_losses = checkpoint['episode_losses']
        return checkpoint  

    def evaluate(self, filepath, num_episodes=1, video_filename='evaluation_video.mp4'):
        self.load_model(filepath)
        frames = []

        for episode in range(num_episodes):
            timestep = self.env.reset()
            episode_reward = 0
            while not timestep.last():
                frame = self.env.physics.render(camera_id=0, width=400, height=304)
                cv2.imshow('Environment', frame)
                cv2.waitKey(1)
                frames.append(frame)

                state = np.concatenate([np.asarray([ob]) if isinstance(ob, float) else ob for ob in timestep.observation.values()], axis=0)
                state = torch.tensor(state).float().to(self.device)
            
                with torch.no_grad():  # No need to track gradients
                    action_mean, action_std, value = self.model(state)
                distribution  = distributions.Normal(action_mean, action_std)
                action = distribution.sample()
                action =torch.clamp(action, min=-1.0, max=1.0)
                env_action = action.clone().detach().cpu().numpy()
                timestep = self.env.step(env_action)
                episode_reward += timestep.reward
            print(f'Episode {episode + 1}, Total Reward: {episode_reward}')

        # Save frames as a video
        self.save_frames_as_video(frames, video_filename)

    def save_frames_as_video(self, frames, video_filename):
        path = os.path.join('.', video_filename)
        imageio.mimsave(path, frames, fps=60)
        print(f'Video saved as {path}')

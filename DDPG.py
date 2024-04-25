import math
import imageio
import numpy as np
import torch
import torch.nn as nn
from collections import deque, namedtuple
import random
import os
import cv2
from tqdm import trange
import numpy as np


Transition = namedtuple('Transition',
                        ('state', 'action', 'reward', 'next_state', 'done')
                        )


class OrnsteinUhlenbeckActionNoise:
    def __init__(self, action_dimension, mu=0, theta=0.15, sigma=0.2): 
        self.action_dim = action_dimension
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dim) * self.mu
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu

    def __call__(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)
    
class Actor(nn.Module):
    def __init__(self, n_states, n_actions, hidden_size=512):
        super(Actor, self).__init__()
        self.l1 = nn.Linear(n_states, hidden_size)
        self.l2 = nn.Linear(hidden_size, 256)
        self.l3 = nn.Linear(256, n_actions)

    def forward(self, state):
        out = torch.relu(self.l1(state))
        out = torch.relu(self.l2(out))
        out = torch.tanh(self.l3(out)) 
        return out
    
class Critic(nn.Module):
    def __init__(self, n_states, n_actions, hidden_size=512):
        super(Critic, self).__init__()
        self.l1 = nn.Linear(n_states + n_actions, hidden_size)
        self.l2 = nn.Linear(hidden_size, 256)
        self.l3 = nn.Linear(256, 1)

    def forward(self, state, action):
        out = torch.cat([state, action], dim=1)
        out = torch.relu(self.l1(out))
        out = torch.relu(self.l2(out))
        out = self.l3(out) 
        return out

class DDPGAgent:
    def __init__(self, env, gamma=0.99, tau=0.005, actor_lr=1e-4, critic_lr=1e-4, 
                 batch_size=1000, memory_size=1000000, seed=42, save_interval=10, save_dir='results'):
        self.seed = seed
        self.set_seed()
        self.env = env
        self.device = torch.device("cuda" if torch.cuda.is_available else "cpu")
        self.gamma = gamma
        self.tau = tau
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.batch_size = batch_size
        self.n_actions = env.action_spec().shape[0]  # 2 for soccer env
        self.n_states = int(sum(np.prod(spec.shape) for spec in env.observation_spec().values()))  # sum(np.prod(spec["shape"]) for spec in env.observation_spec().values())  for soccer env
        self.memory = ReplayBuffer(memory_size)
        self.rewards = [] 
        self.episode_rewards = []
        self.actor_losses = [] 
        self.critic_losses = []
        self.save_interval = save_interval
        self.save_dir = save_dir
        
        # Create Actor network
        self.actor = Actor(self.n_states, self.n_actions).to(self.device)
        self.actor_target = Actor(self.n_states, self.n_actions).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)

        # Create Crictic network
        self.critic = Critic(self.n_states, self.n_actions).to(self.device)
        self.critic_target = Critic(self.n_states, self.n_actions).to(self.device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        
        self.ou_noise = OrnsteinUhlenbeckActionNoise(action_dimension=env.action_spec().shape) # 2 for soccer env

    
    def set_seed(self):
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

    def train(self, num_episodes):
        
        for episode in trange(1,num_episodes+1):
            count = 0
            timestep = self.env.reset()
            self.ou_noise.reset()
            state = np.concatenate([np.asarray([ob]) if isinstance(ob, float) else ob for ob in timestep.observation.values()], axis=0)
            done = False
            episode_reward = 0
            actor_loss, critic_loss = None, None
            while not done:
                count += 1
                with torch.no_grad():
                    action = self.actor(torch.tensor(state).float().to(self.device))
                noise = self.ou_noise()
                action = action.cpu().numpy() + noise
                action = np.clip(action, -1, 1) 

                timestep = self.env.step(action)
                next_state = np.concatenate([np.asarray([ob]) if isinstance(ob, float) else ob for ob in timestep.observation.values()], axis=0) # torch.Tensor([]).cpu()
                reward = timestep.reward
                done = timestep.last()
                self.memory.add(state, action, reward, next_state, done)
                episode_reward += reward
                state = next_state
                if len(self.memory) > 10000:
                    actor_loss, critic_loss = self.update()
                    self.actor_losses.append(actor_loss)
                    self.critic_losses.append(critic_loss)
            self.episode_rewards.append(episode_reward)
            
            print(f"Episode : {episode}, Reward : {episode_reward}, Steps : {count}, Actor Loss : {actor_loss}, Critic Loss : {critic_loss}")
            if episode % self.save_interval == 0:
                self.save_model(os.path.join(self.save_dir, f"ddpg_walker_model_{episode}.pth"))

    def update(self):
        
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))
        state = torch.tensor(np.array(batch.state), dtype=torch.float32).to(self.device)
        action = torch.tensor(np.array(batch.action), dtype=torch.float32).to(self.device)
        reward = torch.tensor(np.array(batch.reward), dtype=torch.float32).to(self.device) 
        next_state = torch.tensor(np.array(batch.next_state), dtype=torch.float32).to(self.device)
        done = torch.tensor(np.array(batch.done), dtype=torch.int32).to(self.device) 

        next_action = self.actor_target(next_state)
        next_action = torch.clamp(next_action, -1, 1) 
        
        next_Q_values = self.critic_target(next_state, next_action)
        next_Q_values = next_Q_values.squeeze()
        expected_Q_values = reward + self.gamma * next_Q_values * (1 - done)
        Q_values = self.critic(state, action) 
        expected_Q_values = expected_Q_values.unsqueeze(1)
        critic_loss = nn.MSELoss()(Q_values, expected_Q_values)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        predicted_action = self.actor(state)
        predicted_action = torch.clamp(predicted_action, -1, 1)
        actor_loss = -torch.mean(self.critic(state, predicted_action))
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.soft_update(self.actor, self.actor_target)
        self.soft_update(self.critic, self.critic_target)
        return actor_loss, critic_loss
    
    def soft_update(self, model, target_model):
        with torch.no_grad():
            for weights, target_weights in zip(model.parameters(), target_model.parameters()):
                target_weights.data.copy_(self.tau * weights.data + (1 - self.tau) * target_weights.data)

    def save_model(self, filename):
        ckpt = {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "actor_target": self.actor_target.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "reply_buffer": self.memory,
            "episode_rewards": self.episode_rewards,
            "actor_losses": self.actor_losses,
            "critic_losses": self.critic_losses
        }
        torch.save(ckpt, filename)
        print("Model saved successfully...")

    def load_model(self, filename):
        ckpt = torch.load(filename)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        self.actor_target.load_state_dict(ckpt["actor_target"])
        self.critic_target.load_state_dict(ckpt["critic_target"])
        self.actor_optimizer.load_state_dict(ckpt["actor_optimizer"])
        self.critic_optimizer.load_state_dict(ckpt["critic_optimizer"])
        self.memory = ckpt["reply_buffer"]
        self.episode_rewards = ckpt["episode_rewards"]
        self.actor_losses = ckpt["actor_losses"]
        self.critic_losses = ckpt["critic_losses"]
        return ckpt

    def render(self, env):
        img = env.physics.render(camera_id=0)  
        cv2.imshow('Environment', img)
        cv2.waitKey(1)

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
            
                with torch.no_grad():  
                    action = self.actor(state)
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
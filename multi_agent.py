import argparse
import os
from matplotlib import pyplot as plt
import torch
from tqdm import trange
from A2C import A2CAgent
from DDPG import DDPGAgent
from SAC import SACAgent
from plots import plot_losses, plot_rewards
from soccer_env import SoccerEnv
import torch.distributions as distributions
import numpy as np


def soccer_train_sac_vs_ddpg(env, agent1, agent2, num_episodes=1000):
    for episode in trange(1,num_episodes+1):
        state = env.reset()
        done = False
        agent1_reward = 0
        agent2_reward = 0
        agent1_saved_rewards = []
        agent2_saved_rewards = []
        actor_loss1, critic_loss1, critic_loss1_1 = None, None, None
        actor_loss2, critic_loss2 = None, None
        agent2.ou_noise.reset()
        count = 0
        while count < 1000:
            count += 1
            
            action1 = get_sac_action(env, agent1, state)
            action2 = get_ddpg_action(env, agent2, state)

            action = torch.cat((action1, action2)).detach().cpu().numpy()
            next_state, reward, done, info = env.step(action)
            agent1.memory.add(state, action1.cpu().numpy(), reward[0], next_state, done)
            agent2.memory.add(state, action2.cpu().numpy(), reward[1], next_state, done)
            agent1_reward += reward[0]
            agent2_reward += reward[1]
            agent1_saved_rewards.append(reward[0])
            agent2_saved_rewards.append(reward[1])
            state = next_state
            if len(agent2.memory) > 10000:
                actor_loss1, critic_loss1, critic_loss1_1 = agent1.update()
                actor_loss2, critic_loss2 = agent2.update()
                agent1.actor_losses.append(actor_loss1)
                agent1.critic1_losses.append(critic_loss1)
                agent1.critic2_losses.append(critic_loss1_1)
                agent2.actor_losses.append(actor_loss2)
                agent2.critic_losses.append(critic_loss2)
            if done:
                state = env.reset()
                break
        agent1.epsiode_rewards.append(agent1_reward)
        agent2.episode_rewards.append(agent2_reward)

        print(f'Episode: {episode}, Steps: {count}, Agent1 Total Reward: {agent1_reward}, Agent2 Total Reward: {agent2_reward}, Actor Loss1: {actor_loss1},  Critic Loss1: {critic_loss1}, Critic Loss1_1: {critic_loss1_1}, Actor Loss2: {actor_loss2}, Critic Loss2: {critic_loss2}')
        if episode % agent1.save_interval == 0:
            agent1.save_model(os.path.join(agent1.save_dir, f"sac_agent1_soccer_model_{episode}.pth"))
            agent2.save_model(os.path.join(agent2.save_dir, f"ddpg_agent2_soccer_model_{episode}.pth"))

# A2C and DDPG agent playing soccer
def soccer_train_a2c_vs_ddpg(env, agent1, agent2, num_episodes=1000):
    for episode in trange(1,num_episodes+1):
        state = env.reset()
        done = False
        agent1_reward = 0
        agent2_reward = 0
        agent1_saved_rewards = []
        agent1_saved_action_values = []
        actor_loss, critic_loss = None, None
        agent2.ou_noise.reset()
        count = 0
        while count < 1000:
            count += 1
            
            action_mean1, action_std1, state_values1 = agent1.model(torch.tensor(state).float().to(agent1.device))
            distribution  = distributions.Normal(action_mean1, action_std1)
            action1 = distribution.sample()
            log_prob1 = distribution.log_prob(action1).sum()
            action1 =torch.clamp(action1, min=-1, max=1) 
            
            action2 = get_ddpg_action(env, agent2, state)

            action = torch.cat((action1, action2)).detach().cpu().numpy()
            next_state, reward, done, info = env.step(action)
            agent2.memory.add(state, action2.cpu().numpy(), reward[1], next_state, done)
            agent1_reward += reward[0]
            agent2_reward += reward[1]
            agent1_saved_rewards.append(reward[0])
            agent1_saved_action_values.append((action1, state_values1, log_prob1))
            state = next_state
            if len(agent2.memory) > 10000:
                actor_loss, critic_loss = agent2.update()
                agent2.actor_losses.append(actor_loss)
                agent2.critic_losses.append(critic_loss)
            if done:
                state = env.reset()
                break
        agent1_loss = agent1.optimize_model(agent1_saved_rewards, agent1_saved_action_values)
        agent1.episode_losses.append(agent1_loss)
        agent1.episode_rewards.append(agent1_reward)
        agent2.episode_rewards.append(agent2_reward)

        print(f'Episode: {episode}, Steps: {count}, Agent1 Total Reward: {agent1_reward}, Agent2 Total Reward: {agent2_reward}, Agent1 Loss: {agent1_loss}, Actor Loss: {actor_loss}, Critic Loss: {critic_loss}')
        if episode % agent1.save_interval == 0:
            agent1.save_model(os.path.join(agent1.save_dir, f"a2c_agent1_soccer_model_{episode}.pth"))
            agent2.save_model(os.path.join(agent2.save_dir, f"ddpg_agent2_soccer_model_{episode}.pth"))

# 2 A2C agents playing soccer
def soccer_train_a2c_vs_a2c(env, agent1, agent2, num_episodes=1000):
    for episode in trange(1,num_episodes+1):
        state = env.reset()
        done = False
        agent1_reward = 0
        agent2_reward = 0
        agent1_saved_rewards = []
        agent2_saved_rewards = []
        agent1_saved_action_values = []
        agent2_saved_action_values = []
        count = 0
        while count < 1000:
            count += 1
            action_mean1, action_std1, state_values1 = agent1.model(torch.tensor(state).float().to(agent1.device))
            distribution  = distributions.Normal(action_mean1, action_std1)
            action1 = distribution.sample()
            log_prob1 = distribution.log_prob(action1).sum()
            action1 = torch.tensor([0,0], dtype=torch.float32).to(agent1.device)
            action1 =torch.clamp(action1, min=-1, max=1) 
            
            action_mean2, action_std2, state_values2 = agent2.model(torch.tensor(state).float().to(agent2.device))
            distribution  = distributions.Normal(action_mean2, action_std2)
            action2 = distribution.sample()
            log_prob2 = distribution.log_prob(action2).sum()
            action2 =torch.clamp(action2, min=-1, max=1) 
            
            action = torch.cat((action1, action2)).detach().cpu().numpy()
            next_state, reward, done, _ = env.step(action)
            agent1_reward += reward[0]
            agent2_reward += reward[1]
            agent1_saved_rewards.append(reward[0])
            agent2_saved_rewards.append(reward[1])
            agent1_saved_action_values.append((action1, state_values1, log_prob1))
            agent2_saved_action_values.append((action2, state_values2, log_prob2))
            state = next_state

            if done:
                state = env.reset()
                break
        agent1_loss = agent1.optimize_model(agent1_saved_rewards, agent1_saved_action_values)
        agent2_loss = agent2.optimize_model(agent2_saved_rewards, agent2_saved_action_values)
        agent1.episode_losses.append(agent1_loss)
        agent2.episode_losses.append(agent2_loss)
        agent1.episode_rewards.append(agent1_reward)
        agent2.episode_rewards.append(agent2_reward)
        
        print(f'Episode: {episode}, Steps: {count}, Agent1 Total Reward: {agent1_reward}, Agent2 Total Reward: {agent2_reward}, Agent1 Loss: {agent1_loss}, Agent2 Loss: {agent2_loss}')
        if episode % agent1.save_interval == 0:
            agent1.save_model(os.path.join(agent1.save_dir, f"a2c_agent1_soccer_model_{episode}.pth"))
            agent2.save_model(os.path.join(agent2.save_dir, f"a2c_agent2_soccer_model_{episode}.pth"))
    

def get_a2c_action(env, agent, state):
    action_mean, action_std, _ = agent.model(torch.tensor(state).float().to(agent.device))
    distribution  = distributions.Normal(action_mean, action_std)
    action = distribution.sample()
    action =torch.clamp(action, min=-1, max=1) 
    return action

#DDPG vs DDPG players
def soccer_train_ddpg_vs_ddpg(env, agent1, agent2, num_episodes=1000):
    for episode in trange(1,num_episodes+1):
        state = env.reset()
        done = False
        agent1_reward = 0
        agent2_reward = 0
        actor_loss1, critic_loss1, actor_loss2, critic_loss2 = None, None, None, None
        agent1.ou_noise.reset()
        agent2.ou_noise.reset()
        count = 0
        while count < 1000:
            count += 1
            action1 = get_ddpg_action(env, agent1, state)
            action2 = torch.tensor([0,0], dtype=torch.float32).to(agent1.device) 

            action = torch.cat((action1, action2)).detach().cpu().numpy()
            next_state, reward, done, info = env.step(action)
            agent1.memory.add(state, action1.cpu().numpy(), reward[0], next_state, done)
            agent2.memory.add(state, action2.cpu().numpy(), reward[1], next_state, done)
            agent1_reward += reward[0]
            agent2_reward += reward[1]
            state = next_state
            
            if len(agent1.memory) > 10000:
                actor_loss1, critic_loss1 = agent1.update()
                actor_loss2, critic_loss2 = agent2.update()
                agent1.actor_losses.append(actor_loss1)
                agent1.critic_losses.append(critic_loss1)
                agent2.actor_losses.append(actor_loss2)
                agent2.critic_losses.append(critic_loss2)
            if done:
                state = env.reset()
                break
        agent1.episode_rewards.append(agent1_reward)
        agent2.episode_rewards.append(agent2_reward)
        
        print(f'Episode: {episode}, Steps: {count}, Agent1 Total Reward: {agent1_reward}, Agent2 Total Reward: {agent2_reward}, \
              Actor Loss1: {actor_loss1}, Critic Loss1: {critic_loss1}, Actor Loss2: {actor_loss2}, Critic Loss2: {critic_loss2}')
        if episode % agent1.save_interval == 0:
            agent1.save_model(os.path.join(agent1.save_dir, f"ddpg_agent1_soccer_model_{episode}.pth"))
            agent2.save_model(os.path.join(agent2.save_dir, f"ddpg_agent2_soccer_model_{episode}.pth"))


def get_ddpg_action(env, agent, state):
    with torch.no_grad():
        action = agent.actor(torch.tensor(state).float().to(agent.device))
    noise = agent.ou_noise()
    action += torch.tensor(noise).to(agent.device)
    action = torch.clamp(action, min=-1, max=1)
    return action

def get_sac_action(env, agent, state):
    with torch.no_grad():
        action, _ = agent.actor.sample(torch.tensor(state).float().unsqueeze(0).to(agent.device))
    action = action.squeeze(0)
    return action

def soccer_eval(env, agent1, agent2, video_filename):
    
    state = env.reset()
    agent1_reward = 0
    agent2_reward = 0
    frames = []
    count = 0
    while count < 1000:
        count += 1
        frame = env.render()
        frames.append(frame)
        
        with torch.no_grad():
            action1 = get_a2c_action(env, agent1, state)
            #action1 = get_sac_action(env, agent1, state)
            action2 = agent2.actor(torch.tensor(state).float().to(agent2.device))
        action1 = torch.clamp(action1, min=-1, max=1)
        action2 = torch.clamp(action2, min=-1, max=1)
        
        action = torch.cat((action1, action2)).detach().cpu().numpy()
        next_state, reward, _, info = env.step(action)
        agent1_reward += reward[0]
        agent2_reward += reward[1]
        state = next_state
    print(f'Agent1 score : {info["agent1_score"]}, Agent2 score : {info["agent2_score"]}')    
    print(f'Agent1 Total Reward: {agent1_reward}, Agent2 Total Reward: {agent2_reward}')
    env.save_frames_as_video(frames, video_filename)
        
def main():
    
    parser = argparse.ArgumentParser(description='Train or evaluate policy on Deepmind Control Suite')
    parser.add_argument('--mode', type=str, choices=['train', 'test'], default='train', help='Mode to run the model: train or test.')
    parser.add_argument('--gamma', type=float, default=0.99, help='Path to the dataset file.')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for the optimizer.')
    parser.add_argument('--episodes', type=int, default=1, help='Number of epochs to train the model.')
    parser.add_argument('--seed', type=int, default=2, help='Random seed for reproducibility.')
    parser.add_argument('--save_interval', type=int, default=2, help='Interval to save the model weights.')
    parser.add_argument('--video_filename', type=str, default='evaluation_cartpole.mp4', help='Filename to save the evaluation video.')
    parser.add_argument('--save_dir', type=str, default='results', help='Path to save the model weights.')
    args = parser.parse_args()
    
    env = SoccerEnv()

    
    agent1 = A2CAgent(env, lr=args.learning_rate, 
                        gamma=args.gamma, 
                        seed=args.seed,
                        save_interval=args.save_interval, 
                        save_dir=args.save_dir)
    
    
    agent2 = DDPGAgent(env, gamma=args.gamma, 
                            seed=42, #args.seed,
                            actor_lr=args.learning_rate,
                            critic_lr=args.learning_rate,
                            batch_size=128, 
                            memory_size=1000000,
                            tau=0.005,
                            save_dir=args.save_dir,
                            save_interval=args.save_interval)
    
    """
    agent1 = SACAgent(env, gamma=args.gamma,
                              seed=2, # args.seed,
                              actor_lr=args.learning_rate,
                              critic_lr=args.learning_rate,
                              batch_size=128,
                              memory_size=1000000,
                              tau=0.005,
                              save_dir=args.save_dir,
                              save_interval=args.save_interval,
                              alpha=0.2,
                              alpha_decay=0.995)
    """
    
    
    if args.mode == 'train':
        soccer_train_a2c_vs_ddpg(env, agent1, agent2, num_episodes=args.episodes)
        #soccer_train_ddpg_vs_ddpg(env, agent1, agent2, num_episodes=args.episodes)
        #soccer_train_sac_vs_ddpg(env, agent1, agent2, num_episodes=args.episodes)
    
    elif args.mode == 'test':
        agent1_ckpt = agent1.load_model("results/a2c_agent1_soccer_model_20.pth")
        agent2_ckpt = agent2.load_model("results/ddpg_agent2_soccer_model_20.pth")
        video_filename = 'videos/a2c_vs_ddpg_evaluation_soccer_20.mp4' 
        soccer_eval(env, agent1, agent2, video_filename)

        # Plots for A2C
        rewards1 = agent1_ckpt['episode_rewards']
        losses1 = agent1_ckpt['episode_losses']
        plot_rewards(rewards1)
        plt.show()
        plot_losses(losses1, "Losses")
        plt.show()
        
        # Plots for DDPG
        rewards = agent2_ckpt['episode_rewards']
        critic_losses = agent2_ckpt['critic_losses']
        actor_losses = agent2_ckpt['actor_losses']
        plot_rewards(rewards)
        plt.show()
        plot_losses(critic_losses, "Critic Losses")
        plt.show()
        plot_losses(actor_losses, "Actor Losses")
        plt.show()
        
        
        """
        # Plots for SAC
        rewards = agent1_ckpt['episode_rewards']
        critic_loss1 = agent1_ckpt['critic_loss_1']
        critic_loss2 = agent1_ckpt['critic_loss_2']
        actor_losses = agent1_ckpt['actor_losses']
        plot_rewards(rewards)
        plt.show()
        plot_losses(critic_loss1, "Critic1 Losses")
        plt.show()
        plot_losses(critic_loss2, "Critic2 Losses")
        plt.show()
        plot_losses(actor_losses, "Actor Losses")
        plt.show()
        """
    
if __name__=="__main__":
    main()

        
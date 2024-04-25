
import matplotlib.pyplot as plt
import torch

def plot_durations(durations: list) -> None:
    durations_t = torch.tensor(durations, dtype=torch.float)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())

def plot_rewards(rewards: list) -> None:
    rewards_t = torch.tensor(rewards, dtype=torch.float)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.plot(rewards_t.numpy())

def plot_losses(losses: list, ylabel) -> None:
    losses_t = torch.tensor(losses, dtype=torch.float)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel(ylabel)
    plt.plot(losses_t.numpy())
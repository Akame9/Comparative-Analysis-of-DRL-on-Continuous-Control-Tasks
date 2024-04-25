import os
import gym
from gym import spaces
import imageio
import numpy as np
import pygame
import sys

class SoccerEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    
    def __init__(self):
        super(SoccerEnv, self).__init__()
        self.field_length = 608
        self.field_width = 400
        self.goal_depth = 40 
        self.goal_width = 80 
        self.goal1_area = (0, (self.field_width - self.goal_width) // 2, self.goal_depth, self.goal_width)
        self.goal2_area = (self.field_length - self.goal_depth, (self.field_width - self.goal_width) // 2, self.goal_depth, self.goal_width)

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        self.observation_space = spaces.Box(low=0, high=np.array([self.field_length, self.field_width]*3), dtype=np.float32)
        
        self.ball_radius = 10
        self.agent_radius = 15  
        self.first_frame_saved = False
        self.reset()
    
    def reset(self):
        self.agent1_score = 0
        self.agent2_score = 0
        agent1_pos = np.array([275, self.field_width / 2])
        agent2_pos = np.array([self.field_length - 275, self.field_width / 2])
        ball_pos = np.array([self.field_length / 2, self.field_width / 2])
        
        self.state = np.concatenate((agent1_pos, agent2_pos, ball_pos))
        return self.state
    
    def distance_to_goal(self, ball_pos, goal_pos):
        return np.sqrt((ball_pos[0] - goal_pos[0])**2 + (ball_pos[1] - goal_pos[1])**2)

    def distance_to_ball(self, agent_pos, ball_pos):
        return np.sqrt((agent_pos[0] - ball_pos[0])**2 + (agent_pos[1] - ball_pos[1])**2)
    
    def step(self, action):
        agent1_pos, agent2_pos, ball_pos = self.state[:2], self.state[2:4], self.state[4:6]
        agent1_ball_to_goal_old_distance = self.distance_to_goal(ball_pos, (self.field_length-self.goal_depth, self.field_width / 2))
        agent2_ball_to_goal_old_distance = self.distance_to_goal(ball_pos, (self.goal_depth, self.field_width / 2))
        agent1_to_ball_old_distance = self.distance_to_ball(agent1_pos, ball_pos)
        agent2_to_ball_old_distance = self.distance_to_ball(agent2_pos, ball_pos)

        agent1_pos += action[:2]
        agent2_pos += action[2:]

        reward = [0, 0]
        done = False
        ball_moved = False
        if np.linalg.norm(agent1_pos - ball_pos) <= self.ball_radius + self.agent_radius:
            ball_pos += action[:2]  
            ball_moved = True
        elif np.linalg.norm(agent2_pos - ball_pos) <= self.ball_radius + self.agent_radius:
            ball_pos += action[2:]
            ball_moved = True
            
        self.state = np.concatenate((agent1_pos, agent2_pos, ball_pos))

        agent1_ball_to_goal_new_distance = self.distance_to_goal(ball_pos, (self.field_length-self.goal_depth, self.field_width / 2))
        agent2_ball_to_goal_new_distance = self.distance_to_goal(ball_pos, (self.goal_depth, self.field_width / 2))
        agent1_to_ball_new_distance = self.distance_to_ball(agent1_pos, ball_pos)
        agent2_to_ball_new_distance = self.distance_to_ball(agent2_pos, ball_pos)
        
        if agent1_to_ball_old_distance > agent1_to_ball_new_distance:
            reward[0] += 10
        else:
            reward[0] -= 5
        if agent2_to_ball_old_distance > agent2_to_ball_new_distance:
            reward[1] += 10
        else:
            reward[1] -= 5
       
        if ball_moved:
            
            if agent1_ball_to_goal_old_distance > agent1_ball_to_goal_new_distance:
                reward[0] += 10
            else:
                reward[0] -= 5
            if agent2_ball_to_goal_old_distance > agent2_ball_to_goal_new_distance:
                reward[1] += 10
            else:
                reward[1] -= 5
            
        if self.goal2_area[0] <= ball_pos[0] <= self.goal2_area[0] + self.goal_depth and self.goal2_area[1] <= ball_pos[1] <= self.goal2_area[1] + self.goal_width:
            self.agent1_score += 1
            print("Goal 1 scored")
            reward[0] += 50
            self.reset()
        if self.goal1_area[0] <= ball_pos[0] <= self.goal1_area[0] + self.goal_depth and self.goal1_area[1] <= ball_pos[1] <= self.goal1_area[1] + self.goal_width:
            self.agent2_score += 1
            print("Goal 2 scored")
            reward[1] += 50
            self.reset()
            
        if ball_pos[0] < 0 or ball_pos[0] > self.field_length \
        or ball_pos[1] < 0 or ball_pos[1] > self.field_width \
        or agent1_pos[0] < 0 or agent1_pos[0] > self.field_length \
        or agent1_pos[1] < 0 or agent1_pos[1] > self.field_width \
        or agent2_pos[0] < 0 or agent2_pos[0] > self.field_length \
        or agent2_pos[1] < 0 or agent2_pos[1] > self.field_width:
            #self.reset()
            reward[0]-= 10
            reward[1]-= 10
        info = {'agent1_score': self.agent1_score, 'agent2_score': self.agent2_score}

        return self.state, reward, done, info
    
    def render(self, mode='human'):
        # Initialize Pygame for rendering
        pygame.init()
        self.screen = pygame.display.set_mode((self.field_length, self.field_width))
        self.clock = pygame.time.Clock()
        self.screen.fill((0, 128, 0))  
        
        # Draw field lines in white
        # Boundary lines
        pygame.draw.rect(self.screen, (255, 255, 255), (0, 0, self.field_length, self.field_width), 5)
        # Center line
        pygame.draw.line(self.screen, (255, 255, 255), (self.field_length // 2, 0), (self.field_length // 2, self.field_width), 5)
        # Center circle
        pygame.draw.circle(self.screen, (255, 255, 255), (self.field_length // 2, self.field_width // 2), 40, 5)
        # Goal areas - Just small rectangles near the goal to mimic real soccer fields (optional, for aesthetics)
        pygame.draw.rect(self.screen, (255, 255, 255), self.goal1_area, 5)
        pygame.draw.rect(self.screen, (255, 255, 255), self.goal2_area, 5)
        
        pygame.draw.circle(self.screen, (0, 0, 255), self.state[:2].astype(int), self.agent_radius)  # Agent 1: Blue
        pygame.draw.circle(self.screen, (255, 0, 0), self.state[2:4].astype(int), self.agent_radius)  # Agent 2: Red
        pygame.draw.circle(self.screen, (125, 125, 125), self.state[4:6].astype(int), self.ball_radius)  # Ball: White
        
        
        pygame.display.flip()
        if not self.first_frame_saved:
            self.save_first_frame()
            self.first_frame_saved = True

        frame = pygame.surfarray.array3d(pygame.display.get_surface())
        frame = frame.transpose([1, 0, 2])
        self.clock.tick(60)  
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
        return frame

    def save_frames_as_video(self, frames, video_filename):
        path = os.path.join('.', video_filename)
        imageio.mimsave(path, frames, fps=60)
        print(f'Video saved as {path}')

    def save_first_frame(self):
        image_path = "first_frame.png"
        pygame.image.save(self.screen, "graphs/first_frame.png")
        print(f"First frame saved to {image_path}")
        
    def observation_spec(self):
        return {
            'agent1_position': {
                'shape': (2,),
                'dtype': np.float32
            },
            'agent2_position': {
                'shape': (2,),
                'dtype': np.float32
            },
            'ball_position': {
                'shape': (2,),
                'dtype': np.float32
            }
        }
    
if __name__ == "__main__":
    env = SoccerEnv()
    env.reset()
    for _ in range(1000):
        env.render()
        env.step(env.action_space.sample())
    env.close()
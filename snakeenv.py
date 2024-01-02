import gymnasium as gym
import numpy as np
from gymnasium import spaces
import numpy as np
import cv2
import random
import time
from collections import deque

SNAKE_LEN_GOAL = 30

def collision_with_apple(apple_position, score):
    apple_position = [random.randrange(1,50)*10,random.randrange(1,50)*10]
    score += 1
    return apple_position, score

def collision_with_boundaries(snake_head):
    if snake_head[0]>=500 or snake_head[0]<0 or snake_head[1]>=500 or snake_head[1]<0 :
        return 1
    else:
        return 0

def collision_with_self(snake_position):
    snake_head = snake_position[0]
    if snake_head in snake_position[1:]:
        return 1
    else:
        return 0
    
def euclidean_distance(a, b):
    return np.sqrt(((a[0] - b[0])**2) + ((a[1] - b[1])**2))

def get_direction_vector(a, b):
    return (b[0] - a[0], b[1] - a[1])

def is_heading_to_apple(snake_head, apple_position):
    direction_vector = get_direction_vector(snake_head, apple_position)
    snake_head_direction = np.array([snake_head[0], snake_head[1]])
    return np.allclose(direction_vector, snake_head_direction)




class SnekEnv(gym.Env):
    """Custom Environment that follows gym interface."""

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self):
        super(SnekEnv, self).__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.truncated = False
        self.action_space = spaces.Discrete(4)
        # Example for using image as input (channel-first; channel-last also works):
        self.observation_space = spaces.Box(low=-500, high=500,
                                            shape=(5 + SNAKE_LEN_GOAL,), dtype=np.float32)

    def step(self, action, terminated=False, truncated=False):

        self.prev_actions.append(action)
        #self.prev_button_direction = action
        cv2.imshow('a',self.img)
        cv2.waitKey(1)
        self.img = np.zeros((500,500,3),dtype='uint8')
        # Display Apple
        cv2.rectangle(self.img,(self.apple_position[0],self.apple_position[1]),(self.apple_position[0]+10,self.apple_position[1]+10),(0,0,255),3)
        # Display Snake
        for position in self.snake_position:
            cv2.rectangle(self.img,(position[0],position[1]),(position[0]+10,position[1]+10),(0,255,0),3)
                                        
        # Takes step after fixed time
        t_end = time.time() + 0.05
        k = -1
        while time.time() < t_end:
            if k == -1:
                k = cv2.waitKey(1)
            else:
                continue
                                                
        # 0-Left, 1-Right, 3-Up, 2-Down, q-Break    
        # a-Left, d-Right, w-Up, s-Down

        button_direction = action
        # Change the head position based on the button direction
        if button_direction == 1:
            self.snake_head[0] += 10
        elif button_direction == 0:
            self.snake_head[0] -= 10
        elif button_direction == 2:
            self.snake_head[1] += 10
        elif button_direction == 3:
            self.snake_head[1] -= 10
            

        # Increase Snake length on eating apple
            
        apple_reward = 0    
        if self.snake_head == self.apple_position:
        # Reposition the apple
            self.apple_position, self.score= collision_with_apple(self.apple_position, self.score)
            self.snake_position.insert(0, list(self.snake_head))
            apple_reward = 10000

        else:
            self.snake_position.insert(0,list(self.snake_head))
            self.snake_position.pop()
                                        
        # On collision kill the snake and print the score
        if collision_with_boundaries(self.snake_head) == 1 or collision_with_self(self.snake_position) == 1:
            font = cv2.FONT_HERSHEY_SIMPLEX
            self.img = np.zeros((500,500,3),dtype='uint8')
            cv2.putText(self.img,'Your Score is {}'.format(self.score),(140,250), font, 1,(255,255,255),2,cv2.LINE_AA)
            cv2.imshow('a',self.img)
            self.done = True

        euclidean_dist_to_apple = euclidean_distance(self.snake_head, self.apple_position)
        if is_heading_to_apple(self.snake_head, self.apple_position):
            self.reward += 0.1

        if self.score == 5:
            self.reward += 100

        self.total_reward = ((250-euclidean_dist_to_apple) + apple_reward)/100

        self.reward = self.total_reward - self.prev_reward
        self.prev_reward = self.total_reward

        if self.done:
            self.reward = -10

        if action == 0:
            self.reward += 0.5
        elif action == 1:
            self.reward += 0.5
        elif action == 2:
            self.reward += 0.1
        elif action == 3:
            self.reward += 0.1
        
        info = {}

        head_x = self.snake_head[0]
        head_y = self.snake_head[1]

        snake_length = len(self.snake_position)
        apple_x = self.apple_position[0] - head_x
        apple_y = self.apple_position[1] - head_y

        self.observation = np.array([head_x, head_y, apple_x, apple_y, snake_length] + list(self.prev_actions), dtype=np.float32)

        return self.observation, self.total_reward, self.done, self.truncated, info

    def reset(self, seed=None):
        self.seedNum = seed
        self.img = np.zeros((500,500,3),dtype='uint8')
        # Initial Snake and Apple position
        self.snake_position = [[250,250],[240,250],[230,250]]
        self.apple_position = [random.randrange(1,50)*10,random.randrange(1,50)*10]
        self.score = 0
        self.prev_button_direction = -1
        self.button_direction = 1
        self.snake_head = [250,250]

        self.prev_reward = 0
        self.done = False

        head_x = self.snake_head[0]
        head_y = self.snake_head[1]

        snake_length = len(self.snake_position)
        apple_x = head_x - self.apple_position[0]
        apple_y = head_y - self.apple_position[1]

        self.prev_actions = deque(maxlen=SNAKE_LEN_GOAL)
        for _ in range(SNAKE_LEN_GOAL):
            self.prev_actions.append(-1)

        self.observation = np.array([head_x, head_y, apple_x, apple_y, snake_length] + list(self.prev_actions), dtype=np.float32)


        info = {}
        return self.observation, info

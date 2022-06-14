import gym_pikachu_volleyball
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym
import random
from collections import deque
import os
from tqdm import tqdm
import argparse
total_rewards = []

parser = argparse.ArgumentParser()
parser.add_argument("--DQN_3000", action="store_true")
parser.add_argument("--DQN_5000", action="store_true")
parser.add_argument("--DQN_20000", action="store_true")
parser.add_argument("--DQN_IMP", action="store_true")
args = parser.parse_args()

class Memory():
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)

    def insert(self, state, action, reward, next_state, done):
        self.memory.append([state, action, reward, next_state, done])

    def sample(self, batch_size):
        batch = random.sample(self.memory, batch_size)
        observations, actions, rewards, next_observations, done = zip(*batch)
        return observations, actions, rewards, next_observations, done

class Net(nn.Module):
    def __init__(self,  num_actions, hidden_layer_size=50):
        super(Net, self).__init__()
        self.input_state = 12  # the dimension of state space
        self.num_actions = num_actions  # the dimension of action space
        self.fc1 = nn.Linear(self.input_state, 32)  # input layer
        self.fc2 = nn.Linear(32, hidden_layer_size)  # hidden layer
        self.fc3 = nn.Linear(hidden_layer_size, num_actions)  # output layer
    
    def forward(self, states):
        x = F.relu(self.fc1(states))
        x = F.relu(self.fc2(x))
        q_values = self.fc3(x)
        return q_values

class Agent():
    def __init__(self, env, epsilon=0.05, learning_rate=0.0002, GAMMA=0.97, batch_size=32, capacity=10000):
        self.env = env
        self.n_actions = 18  # the number of actions
        self.count = 0  # recording the number of iterations

        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.gamma = GAMMA
        self.batch_size = batch_size
        self.capacity = capacity

        self.buffer = Memory(self.capacity)
        self.evaluate_net = Net(self.n_actions)  # the evaluate network
        self.target_net = Net(self.n_actions)  # the target network

        self.optimizer = torch.optim.Adam(
            self.evaluate_net.parameters(), lr=self.learning_rate)  # Adam is a method using to optimize the neural network

def seed(seed=20):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

action_converter_p2 = ((-1, -1,  0),
                       (-1,  0,  0),
                       (-1, +1,  0),
                       (-1, -1, +1),
                       (-1,  0, +1),
                       (-1, +1, +1),
                       ( 0, -1,  0),
                       ( 0,  0,  0),
                       ( 0, +1,  0),
                       ( 0, -1, +1),
                       ( 0,  0, +1),
                       ( 0, +1, +1),
                       (+1, -1,  0),
                       (+1,  0,  0),
                       (+1, +1,  0),
                       (+1, -1, +1),
                       (+1,  0, +1),
                       (+1, +1, +1))

def test(path):
    env = gym.make('gym_pikachu_volleyball:pikachu-volleyball-v0', isPlayer1Computer=True, isPlayer2Computer=False)
    SEED = 17
    seed(SEED)
    env.seed(SEED)
    env.action_space.seed(SEED)

    rewards = []
    win_time, lose_time = [], []
    tt_win = 0
    power_hits = []
    Velocity = []
    testing_agent = Agent(env)
    testing_agent.target_net.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
    # testing_agent1 = Agent(env)
    # testing_agent1.target_net.load_state_dict(torch.load("./Tables/DQN_PLY1.pt", map_location=torch.device('cpu')))

    for _ in range(1000):
        state = env.reset()
        state = state[1]
        # game_done = 0
        win_cnt, lose_cnt, cnt = 0, 0, 0
        time = 0
        ph = 0
        while True:
            time += 1
            Q = testing_agent.target_net.forward(
                torch.FloatTensor(state)).squeeze(0).detach()
            action = int(torch.argmax(Q).numpy())
            next_state, reward, done, _ = env.step([0, action])
            # Q1 = testing_agent1.target_net.forward(
            #     torch.FloatTensor(state)).squeeze(0).detach()
            # action1 = int(torch.argmax(Q1).numpy())
            # next_state, reward, done, _ = env.step([action1, 0])
            next_state = next_state[1]
            move = action_converter_p2[action]
            if move[2] == 1 and abs(env.player2.x - env.ball.x) < 32 and abs(env.player2.y - env.ball.y) < 32:
                ph += 1

            if reward == 1:
                Velocity.append(np.sqrt(env.ball.xVelocity**2 + env.ball.yVelocity**2))
                win_cnt += 1
                win_time.append(time)
            elif reward == -1:
                lose_cnt += 1
                lose_time.append(time)
            elif reward == 0:
                cnt += 1
            if done or cnt > 5000:
                if win_cnt > lose_cnt:
                    tt_win += 1
                rewards.append(win_cnt - lose_cnt)
                power_hits.append(ph)
                break
            state = next_state
    print(f"win time: {np.mean(win_time)}")
    print(f"lose time: {np.mean(lose_time)}")
    print(f"power hits: {np.mean(power_hits)}")
    print(f"velocity: {np.mean(Velocity)}")
    print(f"wins rate: {tt_win/1000}")


if __name__ == "__main__":
    if args.DQN_3000:
        test("./Tables/DQN_3000.pt")
    elif args.DQN_5000:
        test("./Tables/DQN_5000.pt")
    elif args.DQN_20000:
        test("./Tables/DQN_20000.pt")
    elif args.DQN_IMP:
        test("./Tables/DQN_IMP.pt")
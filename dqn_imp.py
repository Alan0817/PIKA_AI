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
total_rewards = []
threshold = 0

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

    def learn(self, ep):
        if self.count % 100 == 0:
            self.target_net.load_state_dict(self.evaluate_net.state_dict())

        observations, actions, rewards, next_observations, done = self.buffer.sample(self.batch_size)
        state = torch.FloatTensor(np.array(observations))
        action = torch.LongTensor(actions)    
        reward = torch.FloatTensor(rewards)   
        next_state = torch.FloatTensor(np.array(next_observations)) 
        done = torch.FloatTensor(np.array(done)) 
        
        state_action_values = self.evaluate_net(state).gather(1, action.unsqueeze(1))
        next_state_values = self.target_net(next_state).max(1)[0].detach()
        expected_state_action_values = (next_state_values * self.gamma) * (1 - done) + reward

        criterion = nn.MSELoss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
        # loss = criterion(state_action_values.view(32), expected_state_action_values)
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.evaluate_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        def test(table):
            env1 = gym.make('gym_pikachu_volleyball:pikachu-volleyball-v0', isPlayer1Computer=True, isPlayer2Computer=False)
            SEED = 17
            env1.seed(SEED)
            env1.action_space.seed(SEED)
            rewards = []
            testing_agent = Agent(env1)
            testing_agent.target_net.load_state_dict(table)
            for _ in range(5):
                state = env1.reset()
                state = state[1]
                # game_done = 0
                cnt = 0
                win_cnt, lose_cnt = 0, 0
                while True:
                    Q = testing_agent.target_net.forward(
                        torch.FloatTensor(state)).squeeze(0).detach()
                    action = int(torch.argmax(Q).numpy())
                    # print(action)
                    next_state, reward, done, _ = env1.step([0, action])
                    next_state = next_state[1]

                    if reward == 1:
                        win_cnt += 1
                    elif reward == -1:
                        lose_cnt += 1
                    elif reward == 0:
                        cnt += 1

                    if done or cnt > 5000:
                        rewards.append(win_cnt - lose_cnt)
                        break
                    state = next_state
            return np.mean(rewards)

        if ep > 2900:
            tmp = test(self.target_net.state_dict())
            global threshold
            if tmp >= threshold:
                threshold = tmp
                torch.save(self.target_net.state_dict(), "./Tables/DQN2.pt")

    def choose_action(self, state):
        with torch.no_grad():
            x = torch.unsqueeze(torch.FloatTensor(state), 0)
            if random.random() < self.epsilon:
                action = env.action_space.sample()
            else:
                actions_value = self.evaluate_net.forward(x)
                # print(torch.max(actions_value, 1))
                action = torch.max(actions_value, 1)[1].data.numpy()
                action = [0, action[0]]
        return action

def train(env):
    agent = Agent(env)
    episode = 3000
    rewards = []
    for ep in tqdm(range(episode)):
        state = env.reset()
        state = state[1]
        game_done = 0
        cnt = 0
        win_cnt, lose_cnt = 0, 0
        r = 0
        while True:
            agent.count += 1
            # env.render()
            action = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            next_state = next_state[1]
            
            if reward == 1:
                win_cnt += 1
                # r1 = 10
                # r += r1
            elif reward == -1:
                lose_cnt += 1
            elif reward == 0:
                cnt += 1

            if win_cnt == 5 or lose_cnt == 5:
                game_done = 1

            if next_state[0] > 0 and not done:
                r1 = (next_state[1]*304/252)**3
                r += r1
            elif next_state[0] < 0 and not done:
                r1 = -1*(next_state[1]*304/252)**15
                r += r1
            else:
                r1 = 0
            # elif next_state[2] > 0 and next_state[3] > 0:
            #     vel = np.sqrt(next_state[2]**2 + next_state[3]**2)
            #     r *= (1 - vel)**()
            
            if reward == -1:
                r += reward
            agent.buffer.insert(state, action[1], r1, next_state, int(game_done))

            if agent.count >= 1000:
                agent.learn(ep)
            if done or cnt > 5000:
                rewards.append(r)
                break
            state = next_state
    total_rewards.append(rewards)

def test(env):
    rewards = []
    testing_agent = Agent(env)
    testing_agent.target_net.load_state_dict(torch.load("./Tables/DQN2.pt"))
    for _ in range(100):
        state = env.reset()
        state = state[1]
        # game_done = 0
        win_cnt, lose_cnt, cnt = 0, 0, 0
        while True:
            Q = testing_agent.target_net.forward(
                torch.FloatTensor(state)).squeeze(0).detach()
            action = int(torch.argmax(Q).numpy())
            next_state, reward, done, _ = env.step([0, action])
            next_state = next_state[1]

            if reward == 0:
                cnt += 1
            if done or cnt > 5000:
                rewards.append(reward)
                break
            state = next_state
    print(f"reward: {np.mean(rewards)}")
    # print(f"max Q:{testing_agent.check_max_Q()}")

def seed(seed=20):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

if __name__ == "__main__":
    env = gym.make('gym_pikachu_volleyball:pikachu-volleyball-v0', isPlayer1Computer=True, isPlayer2Computer=False)
    SEED = 17
    seed(SEED)
    env.seed(SEED)
    env.action_space.seed(SEED)
    # env.reset()
    # env.render()

    if not os.path.exists("./Tables"):
        os.mkdir("./Tables")

    # training section:
    for i in range(1):
        print(f"#{i + 1} training progress")
        train(env)
    # testing section:
    test(env)

    if not os.path.exists("./Rewards"):
        os.mkdir("./Rewards")

    np.save("./Rewards/DQN_rewards2.npy", np.array(total_rewards))

    env.close()
    
import numpy as np
import gym
import os
from tqdm import tqdm
import time
import random

total_reward = []
episode = 3000
decay = 0.045

testCount = 0
LastTestCount = 0
trainTimes = 0
LastReward = 0

class Agent():
    def __init__(self, env, epsilon=0.05, learning_rate=0.5, GAMMA=0.97, num_bins=3):
        
        self.env = env

        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.gamma = GAMMA

        self.num_bins = num_bins
        self.qtable = np.zeros((self.num_bins, 
                                self.num_bins,
                                self.num_bins, 
                                self.num_bins,
                                self.num_bins, 
                                self.num_bins,
                                self.num_bins, 
                                self.num_bins,
                                self.num_bins, 
                                self.num_bins,
                                self.num_bins, 
                                self.num_bins, 18))

        
        self.bins = [
            self.init_bins(-1, 1, self.num_bins),  
            self.init_bins(0, 1, self.num_bins),  
            self.init_bins(-1, 1, self.num_bins),
            self.init_bins(-1, 1, self.num_bins),  
            self.init_bins(-1, 0, self.num_bins),  
            self.init_bins(0, 1, self.num_bins),
            self.init_bins(-1, 1, self.num_bins),  
            self.init_bins(-1, 1, self.num_bins),  
            self.init_bins(0, 1, self.num_bins),
            self.init_bins(0, 1, self.num_bins),  
            self.init_bins(-1, 1, self.num_bins),  
            self.init_bins(-1, 1, self.num_bins),
            
        ]

    def init_bins(self, lower_bound, upper_bound, num_bins):
        
        
        interval = np.linspace(lower_bound, upper_bound, num_bins + 1 )
        modifiedInterval = np.delete(interval, 0)
        modifiedInterval = np.delete(modifiedInterval, -1)
        
        return modifiedInterval
       

    def discretize_value(self, value, bins):
        
        return np.digitize(value, bins) 

       

    def discretize_observation(self, observation):
        
        
        features = []
        for i in range(len(observation[1])):
            features.append(self.discretize_value(observation[1][i],self.bins[i]))
        return tuple(features)
       

    def choose_action(self, state):
        
        
        randomValue= np.random.rand()
        
        if randomValue < self.epsilon:
            action = env.action_space.sample()
            
        else:
            myAction = np.argmax(self.qtable[state])
            action = [random.randrange(0, 17),myAction]

                      
        return action
        

    def learn(self, state, action, reward, next_state, done):
        
        key = 0
        
        
        if done == True:
            self.qtable[state][action[1]] = (1 - self.learning_rate)*(self.qtable[state][action[1]]) + self.learning_rate*(reward + self.gamma*(0))
            global testCount
            global trainTimes
            global LastReward
            
            testCount = 0
            trainTimes = trainTimes + 1
            
        else :
            testCount = testCount + 1
            new_qtable = (1 - self.learning_rate)*(self.qtable[state][action[1]]) + self.learning_rate*(reward + self.gamma*(np.max(self.qtable[next_state])))
            
            self.qtable[state][action[1]] = new_qtable
            
        
       
        if (trainTimes%3000 > 2995 and done == True):
           
            np.save("./Tables/pika_q_table_3000.npy", self.qtable)
           
                
            

    
        
def train(env):
    
    training_agent = Agent(env)
    rewards = []
    
    
    for ep in tqdm(range(episode)):
        state = training_agent.discretize_observation(env.reset())
        done = False
        init_reward = 0
        count = 0
        while True:
            
            # env.render()
            count += 1
            
            action = training_agent.choose_action(state)
            
            r = 0
            next_observation, reward, done, _ = env.step(action)
            
            if(next_observation[1][0]> 0):
                r  = (next_observation[1][1]*304/252)**3

            reward = init_reward + reward*10 + r
            
            
            if count > 1000 :
                print(reward)
                break
            next_state = training_agent.discretize_observation(
                next_observation)
            
            training_agent.learn(state, action, reward, next_state, done)
           
            
            if done:
                
                rewards.append(reward)
                break

            state = next_state


        if (ep + 1) % 500 == 0:
            training_agent.learning_rate -= decay

    total_reward.append(rewards)


def test(env):
    
    testing_agent = Agent(env)

   
    testing_agent.qtable = np.load("./Tables/pika_q_table_3000.npy")
    rewards = []
    winTimes=[]
    loseTimes=[]
    powHitTimes=[]
    velocitys=[]
    for _ in range(1000):
        state = testing_agent.discretize_observation(testing_agent.env.reset())
        count = 0
        powHitTime = 0
        while True:
            
            velocity = 0
            myAction = np.argmax(testing_agent.qtable[tuple(state)])
            
            count +=1
            action = [random.randrange(0, 17),myAction]
            next_observation, reward, done, _ = testing_agent.env.step(action)

            
            if(myAction==3 or 4 or 5 or  9 or 10 or 11 or  15 or 16 or 17):
                if(abs(env.player2.x- env.ball.x)<32 and abs(env.player2.x- env.ball.x)<32):
                    powHitTime +=1

            next_state = testing_agent.discretize_observation(next_observation)


            if done == True:
                
                if reward == 1:
                    
                    print("WIN")
                    
                    velocity = ((env.ball.xVelocity)**2 + (env.ball.xVelocity)**2)**0.5
                    velocitys.append(velocity)
                    winTimes.append(count)
                    powHitTimes.append(powHitTime)
                    rewards.append(1)
                else :
                    print("LOSE")
                    
                    loseTimes.append(count)
                    powHitTimes.append(powHitTime)
                    rewards.append(0)
                break

            state = next_state

    print(f"Win times: {np.mean(winTimes)}")
    print(f"Lose times: {np.mean(loseTimes)}")
    print(f"Power hit times: {np.mean(powHitTimes)}")
    print(f"Velocity: {np.mean(velocitys)}")
    print(f"Win: {np.sum(rewards)}")
   


def seed(seed=20):
    
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)


if __name__ == "__main__":
    
    SEED = 20

    env = gym.make('gym_pikachu_volleyball:pikachu-volleyball-v0', isPlayer1Computer=True, isPlayer2Computer=False)
    seed(SEED)
    env.seed(SEED)
    env.action_space.seed(SEED)

    if not os.path.exists("./Tables"):
        os.mkdir("./Tables")

   # training section:
    for i in range(1):
        print(f"#{i + 1} training progress")
        
        #train(env)
    #testing section:
    test(env)

    if not os.path.exists("./Rewards"):
        os.mkdir("./Rewards")

    np.save("./Rewards/pika_q_rewards_3000.npy", np.array(total_reward))

    env.close()

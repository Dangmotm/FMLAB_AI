import numpy as np

class environment:
    
    def __init__(self, grid_height, grid_width):

        self.height = grid_height
        self.width = grid_width
        self.start = []
        self.end = []
        self.reward = []
        self.map = np.array([i for i in range(grid_height * grid_width)])
        self.action_space = [0, 1, 2, 3]

    def get_Map(self):
        print(self.map.reshape([self.width, self.height]))

    def get_NumState(self):
        return self.height * self.width

    def map_Designate(self, start_cell, end_cell, reward):
        self.start.append(start_cell)
        self.end.append(end_cell)
        self.reward.append(reward)

    def get_Observation(self, location, action):
        if action == -1:
            return None, self.action_space, None

        new_location = 0

        if location in self.start:
            idx = self.start.index(location)
            new_location = self.end[idx]
            reward = self.reward[idx]
            return new_location, self.action_space, reward

        reward = 0

        if action == 0: #UP
            if location - self.width < 0:
                new_location = location
            else:
                new_location = location - self.width

        elif action == 1: #DOWN
            if location + self.width > len(self.map) - 1:
                new_location = location
            else:
                new_location = location + self.width

        elif action == 2: #LEFT
            if location % self.width == 0:
                new_location = location
            else:
                new_location = location - 1

        elif action == 3: #RIGHT
            if (location + 1) % self.width == 0:
                new_location = location
            else:
                new_location = location + 1

        return new_location, self.action_space, reward

Envir = environment(8, 8)
Envir.get_Map()
Envir.map_Designate(17, 56, -15)
Envir.map_Designate(18, 56, -15)
Envir.map_Designate(19, 56, -15)
Envir.map_Designate(21, 56, -15)
Envir.map_Designate(25, 56, -15)
Envir.map_Designate(33, 56, -15)
Envir.map_Designate(41, 56, -15)
Envir.map_Designate(42, 56, -15)
Envir.map_Designate(43, 56, -15)
Envir.map_Designate(46, 56, -15)
Envir.map_Designate(47, 56, -15)
Envir.map_Designate(47, 56, -15)
Envir.map_Designate(15, 56, +15)
Envir.map_Designate(1, 10, +5)
Envir.map_Designate(26, 56, +20)

for i in range(len(Envir.start)):
    print('i = ' + str(i) + ' [start at ' + str(Envir.start[i]) + ' results at '
          + str(Envir.end[i]) + ' get Reward: ' + str(Envir.reward[i]))

a, b, r = Envir.get_Observation(1, 0)
print(r)

toss = np.random.rand()
toss

class MAB_agent:
    def __init__(self, envir, init_location):
        self.reward_trace = []
        self.location_now = init_location
        self.lastAction = None
        self.lastState = None
        self.value_table = {}

    def get_TotalReward(self):
        return np.sum(self.reward_trace)

    def getAction(self, observation):
        self.location_now, action_space, pre_reward = observation

        if self.location_now not in self.value_table.keys():
            self.value_table[self.location_now] = {i : [0, 1] for i in action_space}

        if pre_reward is None:
            action = np.random.choice(action_space, p = [1 / (len(action_space)) for action in action_space])
        else:
            self.reward_trace.append(pre_reward)

            value = self.value_table[self.lastState][self.lastAction][0]
            count = self.value_table[self.lastState][self.lastAction][1]

            count += 1
            value += (1 / count) * (pre_reward - value)

            self.value_table[self.lastState][self.lastAction][0] = value
            self.value_table[self.lastState][self.lastAction][1] = count

            state_dict = self.value_table[self.lastState].values()
            state_dict_array = np.array(list(state_dict))
            value_column = state_dict_array[:, 0]
            action = np.argmax(value_column)

        self.lastState = self.location_now
        self.lastAction = action

        assert action in action_space, "INVALID action taken"
        return action

class MABe_agent(MAB_agent):
    def __init__(self, envir, init_location, epsilon):
        
        super(MABe_agent, self).__init__(envir, init_location)
        self.epsilon = epsilon

    def getAction(self, observation):
        self.location_now, action_space, pre_reward = observation

        if self.location_now not in self.value_table.keys():
            self.value_table[self.location_now] = {i : [0, 1] for i in action_space}

        toss = np.random.rand()
        
        if pre_reward is None or toss < self.epsilon:
            action = np.random.choice(action_space, p = [1 / (len(action_space)) for action in action_space])
        else:
            self.reward_trace.append(pre_reward)

            value = self.value_table[self.lastState][self.lastAction][0]
            count = self.value_table[self.lastState][self.lastAction][1]

            count += 1
            value += (1 / count) * (pre_reward - value)

            self.value_table[self.lastState][self.lastAction][0] = value
            self.value_table[self.lastState][self.lastAction][1] = count

            state_dict = self.value_table[self.lastState].values()
            state_dict_array = np.array(list(state_dict))
            value_column = state_dict_array[:, 0]
            action = np.argmax(value_column)

        self.lastState = self.location_now
        self.lastAction = action

        assert action in action_space, "INVALID action taken"
        return action

init_location = 0
dummyAgent = MAB_agent(envir = Envir, init_location = init_location)

num_iter = 100
log_freq = 10
Data_plot1 = []

for i in range(num_iter):
  env_observation = (init_location, Envir.action_space, None)
    
  if i > 0:
    env_observation = Envir.get_Observation(location = dummyAgent.location_now, action = chosen_action)

  chosen_action = dummyAgent.getAction(observation = env_observation)
    
  if (i + 1) % log_freq == 0:
    aver = np.mean(dummyAgent.reward_trace)
    Data_plot1.append(aver)
    print('iter: ' + str(i + 1) + '\t Total reward: ' + str(dummyAgent.get_TotalReward()) + '\t Average: ' + str(aver))

init_location = 0
epsilon = 0.5
dummyAgent = MABe_agent(envir = Envir, init_location = init_location, epsilon = epsilon)

num_iter = 100
log_freq = 10
Data_plot2 = []

for i in range(num_iter):
  env_observation = (init_location, Envir.action_space, None)
    
  if i > 0:
    env_observation = Envir.get_Observation(location = dummyAgent.location_now, action = chosen_action)

  chosen_action = dummyAgent.getAction(observation = env_observation)
    
  if (i + 1) % log_freq == 0:
    aver = np.mean(dummyAgent.reward_trace)
    Data_plot2.append(aver)
    print('iter: ' + str(i + 1) + '\t Total reward: ' + str(dummyAgent.get_TotalReward()) + '\t Average: ' + str(aver))

import matplotlib.pyplot as plt

fig = plt.figure()
plt.plot(Data_plot1, label = "No exploration")
plt.plot(Data_plot2, label = "e-greedy")
plt.legend()
plt.show()
import random
import numpy as np
#  random stuff, from chatgpt lmao. 
class ActorCriticAgent:
    def __init__(self, state_size, action_size, learning_rate, discount_factor):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.actor = np.zeros((state_size, action_size))
        self.critic = np.zeros(state_size)

    def get_action(self, state):
        state_value = self.critic[state]
        policy = np.exp(self.actor[state] - np.max(self.actor[state])) / np.sum(np.exp(self.actor[state] - np.max(self.actor[state])))
        action = np.random.choice(self.action_size, p=policy)
        return action, state_value

    def learn(self, state, action, reward, next_state, next_state_value, game_over):
        state_value = self.critic[state]
        advantage = reward + self.discount_factor * next_state_value * (1 - game_over) - state_value
        self.actor[state] += self.learning_rate * advantage * (np.exp(self.actor[state][action] - np.max(self.actor[state])) / np.sum(np.exp(self.actor[state] - np.max(self.actor[state]))))
        self.critic[state] += self.learning_rate * advantage
        
def step(state, action):
    # update state based on action
    if action == 0:
        next_state = state - 1
    elif action == 1:
        next_state = state + 1
    else:
        next_state = state
    
    # determine reward
    if next_state == 5:
        reward = 100
        game_over = True
    elif next_state < 0 or next_state > 9:
        reward = -100
        game_over = True
        next_state = state
    else:
        reward = -1
        game_over = False
        
    return next_state, reward, game_over


def play_game(agent, max_steps=10):
    state = random.randint(0, 9)
    for s in range(max_steps):
        action, state_value = agent.get_action(state)
        next_state, reward, game_over = step(state, action)
        next_state_value = agent.critic[next_state]
        agent.learn(state, action, reward, next_state, next_state_value, game_over)
        state = next_state
        if game_over:
            break
    return s

agent = ActorCriticAgent(10, 2, 0.1, 0.99)
steps = [play_game(agent) for episode in range(100)]


import matplotlib.pyplot as plt
import numpy as np

def plot_grid_and_agent(agent, max_steps=10):
    state = random.randint(0, 9)
    states = [state]
    fig, ax = plt.subplots()
    ax.matshow(np.zeros((3,3)), cmap='gray')
    for s in range(max_steps):
        action, state_value = agent.get_action(state)
        next_state, reward, game_over = step(state, action)
        states.append(next_state)
        state = next_state
        x, y = np.unravel_index(state, (3, 3))
        ax.scatter(y, x, c='red', s=100)
        if game_over:
            break
    plt.show()

plot_grid_and_agent(agent, 10)

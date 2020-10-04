#!/usr/bin/env python
import click
import gym
import numpy as np 
import sys 
from collections import defaultdict, deque
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt 

gym.envs.register(
     id='gridworld-v0',
     entry_point='rlpa2:gridworld',
     max_episode_steps=150,
     # kwargs={'size' : 1, 'init_state' : 10., 'state_bound' : np.inf},
)

def update_sarsa(alpha, gamma, Q,state, action, reward, next_state=None, next_action = None, done=False ):
	current_sa = Q[state[0],state[1]][action]
	target_sa = reward + (gamma*Q[next_state[0],next_state[1]][next_action])
	Q[state[0],state[1]][action] = current_sa + alpha*(target_sa - current_sa)
	return Q

def epsilon_greedy(Q, state, nA, epsilon):
	if np.random.random() > epsilon:
		return np.argmax(Q[state[0],state[1],:])
	else:
		return np.random.choice(np.arange(env.action_space.n))

def sarsa(env,goal, num_episodes, alpha,eps=0.1, gamma=1.0, plot_every=100):
	goal = env.goal(goal)
	nA = env.action_space               # number of actions
	Q = np.random.rand(env.observation_space.shape[0], env.observation_space.shape[1], env.action_space.n) # initialize array for Q
	steps, rewards, epis = [], [], [] 
	for episode in range(1, num_episodes+1):
		epis.append(episode)
		# if(episode%100==0):
		# 	print("episode",episode)
		ep_rew, ep_step = 0.0, 0
		state = env.reset()                                   # start episode
		# eps = 1.0 / episode                                 # set value of epsilon
		action = epsilon_greedy(Q, state, nA, eps)            # epsilon-greedy action selection

		while True:
			next_state, reward, done, _ = env.step(action) # take action A, observe R, S'
			ep_rew += reward   
			ep_step += 1   
			if not done:
				next_action = epsilon_greedy(Q, next_state, nA, eps) # epsilon-greedy action
				Q = update_sarsa(alpha, gamma, Q, state, action, reward, next_state, next_action, done)
				state = next_state     
				action = next_action  
			if done:
				Q[state[0],state[1]][action]  = (1 - alpha)*Q[state[0],state[1]][action] + alpha*reward 
				rewards.append(ep_rew)    # append score
				steps.append(ep_step)
				break
	return Q, epis, rewards, steps

goal = "B"
gamma = 0.9
alpha_ls = [0.01, 0.025, 0.05, 0.075, 0.09, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99]
epsilon_ls = [0.01, 0.025, 0.05, 0.075, 0.09, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99]
alpha = 0.01
epsilon = 0.1
num_of_episodes = 20000
avg_rewards = [0]*num_of_episodes
avg_steps = [0]*num_of_episodes

# for alpha in alpha_ls:
# 	for epsilon in epsilon_ls:
env = gym.make('gridworld-v0')
print(env.action_space) # 0 - LEFT, 1 - RIGHT, 2 - UP, 3 -DOWN
print(env.observation_space) # [12, 12] 2D array

for iter in range(50):
	if(iter%10==0):
		print("iter",iter)
	env = gym.make('gridworld-v0')
	Q_sarsa, episode_ls, rewards_ls, steps_ls = sarsa(env, goal,num_of_episodes,alpha,epsilon ,gamma)

	avg_rewards = [x+y for x,y in zip(avg_rewards, rewards_ls)] 
	avg_steps = [x+y for x,y in zip(avg_steps, steps_ls)] 

print("avg_rewards",avg_rewards)
print("avg_steps",avg_steps)
avg_rewards[:] = [x / 50 for x in avg_rewards]
avg_steps[:] = [x / 50 for x in avg_steps]
# plt.plot(episode_ls, rewards_ls) 
plt.plot(episode_ls, avg_rewards) 
plt.xlabel('number of episodes') # X axis label
plt.ylabel('reward per episode') # Y axis label
# plt.yticks(np.arange(min(avg_rewards), max(avg_rewards)+10, 10.0))
plt.title("SARSA average rewards over 50 iterations for goal " + str(goal)) # Show legend
plt.show() # show the plot
plt.close() # Close the plot

# plt.plot(episode_ls, steps_ls) 
plt.plot(episode_ls, avg_steps) 
plt.xlabel('number of episodes') # X axis label
plt.ylabel('steps per episode') # Y axis label
# plt.yticks(np.arange(min(avg_steps)-10, max(avg_steps)+10, 10.0))
plt.title("SARSA average steps over 50 iterations for goal " + str(goal)) # Show legend
plt.show() # show the plot
plt.close() # Close the plot
policy = np.zeros([12,12])
for i in range(12):
	for j in range(12):
		policy[i,j] = np.argmax(np.array(Q_sarsa)[i,j])
		print(np.argmax(np.array(Q_sarsa)[i,j]))
print(policy)
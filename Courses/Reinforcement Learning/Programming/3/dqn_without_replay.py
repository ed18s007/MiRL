# https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
import gym 
import math 
import random 
import numpy as np 
import matplotlib.pyplot as plt 
from collections import namedtuple, deque
from itertools import count 
from PIL import Image 
import os
os.environ['LANG']='en_US'

import torch 
import torch.nn as nn 
import torch.optim as optim
import torch.functional as F 
import torchvision.transforms as T 

# seed_val = 1234
# random.seed(seed_val)
# np.random.seed(seed_val)
# torch.manual_seed(seed_val)

env = gym.make('CartPole-v0')
l = env.reset()
print(l)
plt.ion()

device = "cpu"

Transition = namedtuple('Transition',
	('state','action', 'reward', 'done', 'next_state'))


class DQN(nn.Module):
	def __init__(self, input_shape, num_actions):
		super(DQN, self).__init__()
		self.pipeline = nn.Sequential(
			nn.Linear(input_shape[0], HIDDEN1_SIZE),
			nn.ReLU(),
			# nn.Linear(HIDDEN1_SIZE, HIDDEN2_SIZE),
			# nn.ReLU(),
			nn.Linear(HIDDEN1_SIZE, num_actions)
			)

	def forward(self, x):
		return self.pipeline(x)

REPLAY_MEMORY_SIZE = 50000 			# number of tuples in experience replay  
HIDDEN1_SIZE = 128 					# size of hidden layer 1
EPISODES_NUM = 1000		        	# number of episodes to train on. Ideally shouldn't take longer than 2000
LEARNING_RATE = 0.01  				# learning rate and other parameters for SGD/RMSProp/Adam
BATCH_SIZE = 128     				# size of minibatch sampled from the experience replay
GAMMA = 0.99         				# MDP's gamma
TARGET_UPDATE_FREQ = 8  			# number of steps (not episodes) after which to update the target networks 
TOTAL_STEPS = 0
MEAN_REWARD_BOUND = 195.0
REPLAY_START_SIZE = 10000
EPSILON = 0.1

n_actions = env.action_space.n
n_inputs = env.observation_space.shape
policy_net = DQN(n_inputs,n_actions)
target_net = DQN(n_inputs,n_actions)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.SGD(policy_net.parameters(),lr = LEARNING_RATE)
# optimizer = optim.Adam(policy_net.parameters(),lr = LEARNING_RATE)
# optimizer = optim.RMSprop(policy_net.parameters())
print(policy_net)
steps_done = 0

def select_Action(state):
	global steps_done
	sample = random.random()
	steps_done +=1
	if sample>EPSILON:
		# print("eps_threshold",eps_threshold)
		with torch.no_grad():
			state_a = np.array([state], copy=True)
			state_t = torch.tensor(state_a).to(device)
			q_vals_v = policy_net(state_t.float())
			_, act_v = torch.max(q_vals_v, dim=1)
			action = int(act_v.item())
	else:
		action =  env.action_space.sample()
	return action

episode_durations = []
save = False
def plot_durations(save):
	plt.figure(2)
	plt.clf()
	durations_t = torch.tensor(episode_durations, dtype=torch.float)
	plt.title('Without Experience Replay')
	plt.xlabel('Episode')
	plt.ylabel('Duration')
	
	plt.plot(durations_t.numpy(), label = "Rewards")
	# Take 100 episode averages and plot them too
	if len(durations_t) >= 100:
		means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
		means = torch.cat((torch.zeros(99), means))
		plt.plot(means.numpy(), label = "Average over 100 episodes")
	plt.legend()
	if save:
		plt.savefig('plts/Figure_'+ str(LEARNING_RATE)+'.png')
	plt.pause(0.001)  # pause a bit so that plots are updated

episode = 0
while episode <EPISODES_NUM:
	state = env.reset()
	t= 0
	state_ls, action_ls, reward_ls, done_ls, next_state_ls = [], [], [], [], []
	while t<BATCH_SIZE:
		t += 1
		TOTAL_STEPS += 1
		action = select_Action(state)
		next_state, reward, done, _ = env.step(action)
		state_ls.append(state)
		action_ls.append(action)
		reward_ls.append(reward)
		done_ls.append(done)
		next_state_ls.append(next_state)
		if done:
			episode += 1
			state = env.reset()
			episode_durations.append(t + 1)
			plot_durations(save)
			max_rew = max(episode_durations)
			mean_rewards = float(np.mean(episode_durations[-100:]))
			# Update the target network, copying all weights and biases in DQN
			if episode%TARGET_UPDATE_FREQ==0:
				target_net.load_state_dict(policy_net.state_dict())

			if episode==(EPISODES_NUM-1) or mean_rewards>=195:
				print("Solved in %d total steps in %d episodes!" % (TOTAL_STEPS,episode))
				save=True
			states_t = torch.tensor(np.array(state_ls)).to(device)
			actions_t = torch.tensor(action_ls).to(device)
			rewards_t = torch.tensor(reward_ls).to(device)
			dones_mask_t = torch.BoolTensor(done_ls).to(device)
			next_states_t = torch.tensor(np.array(next_state_ls)).to(device)

			state_action_values = policy_net(states_t.float()).gather(1, actions_t.unsqueeze(-1)).squeeze(-1)
			next_states_vals = target_net(next_states_t.float()).max(1)[0]
			next_states_vals[dones_mask_t] = 0.0
			next_states_vals = next_states_vals.detach()
			expected_state_action_values = GAMMA*next_states_vals + rewards_t
			loss = nn.MSELoss()(state_action_values, expected_state_action_values)
			# loss = nn.SmoothL1Loss()(state_action_values, expected_state_action_values)

			optimizer.zero_grad()
			loss.backward()
			for param in policy_net.parameters():
				param.grad.data.clamp_(-1, 1)
			optimizer.step()
			break



print('Complete')
env.render()
env.close()
plt.ioff()
plt.show()
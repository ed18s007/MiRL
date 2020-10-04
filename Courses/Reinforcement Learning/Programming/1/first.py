# sample-average method for estimating action values
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import random 

x_values =  np.arange(1, 1001, 1) 
k = 10

################################################################################################################
################################################################################################################
def epsilon_greedy(epsilon, average_reward, opt_action, mean_dist,k=10):
	q_est_t_a = [0]*k
	num_actions = [0]*k
	rewards_ls = [0]*k
	avg_r = 0
	mean_opt_action = np.argmax(mean_dist)
	for i in range(1000):
		# A_t = argmax Qt(a)
		A_t = np.argmax(q_est_t_a, axis=0)
		# select uniformly between all k actions
		uniform_choice = np.random.randint(k, size=1)	
		# choose with a probability of epsilon uniformly and other greedily
		action = int(np.random.choice( [A_t, uniform_choice], 1, p=[1-epsilon, epsilon]))
		# update action taken in num action
		num_actions[action] = num_actions[action] + 1
		# choose from normal distribution a reward with mu and sigma as below
		sigma = 1
		mu = mean_dist[action]
		reward = float(np.random.normal(mu, sigma, 1))
		average_reward[i] += reward
		# update reward in reward list
		# rewards_ls[action] = rewards_ls[action] + reward
		# update action value in Q estimated at time t action a
		q_est_t_a[action] += (reward- q_est_t_a[action])/num_actions[action]
		if (action==mean_opt_action):
			opt_action[i] +=1
	return average_reward, opt_action

# epsilons = [0, 0.01, 0.03, 0.08, 0.1]
# opt_action_ls = []
# avg_reward_ls = []
# for i in range(len(epsilons)):
# 	avg_reward_ls.append([0]*1000)
# 	opt_action_ls.append([0]*1000)
# num_times = 2000
# for i in range(num_times):
# 	mean_dist = np.random.normal(0, 1, 10)
# 	y_values = []	
# 	for e in range(len(epsilons)):
# 		avg_reward_ls[e], opt_action_ls[e] = epsilon_greedy(epsilons[e], avg_reward_ls[e], opt_action_ls[e],mean_dist,10)
# for j in range(len(epsilons)):
# 	avg_reward_ls[j] = [ x / num_times for x in avg_reward_ls[j]]
# 	plt.plot(x_values, avg_reward_ls[j], label = "\u03B5 ="+str(epsilons[j])) 
# plt.xlabel('Steps') 
# plt.ylabel('Average reward') 
# plt.legend() 
# plt.show() 
# plt.close()

# for j in range(len(epsilons)):
# 	opt_action_ls[j] = [(x*100) / num_times for x in opt_action_ls[j]]
# 	plt.plot(x_values, opt_action_ls[j], label = "\u03B5 ="+str(epsilons[j])) 
# 	plt.ylim(0.0, 100.0)
# plt.xlabel('Steps') 
# plt.ylabel('%  optimal action') 
# plt.legend() 
# plt.show() 
# plt.close()

################################################################################################################
################################################################################################################
def np_softmax(x):
	e_x = np.exp(x)
	return e_x / e_x.sum()

def softmax(temperature, average_reward,opt_action, mean_dist,k=10):
	q_est_t_a = [0]*k
	num_actions = [0]*k
	rewards_ls = [0]*k
	avg_r = 0
	act_ls =  np.arange(k)
	mean_opt_action = np.argmax(mean_dist)
	for i in range(1000):
		# act_prob_dist = np_softmax(q_est_t_a)
		exp_A_t =   [np.exp(x / temperature) for x in q_est_t_a]
		exp_A_t = np.clip(exp_A_t, a_min =1e-20, a_max = 1e+20) 
		act_prob_dist = exp_A_t/np.sum(exp_A_t)
		action = int(np.random.choice(act_ls, 1, p=act_prob_dist))
		# update action taken in num action
		num_actions[action] = num_actions[action] + 1
		# choose from normal distribution a reward with mu and sigma as below
		sigma = 1
		mu = mean_dist[action]
		reward = float(np.random.normal(mu, sigma, 1))
		average_reward[i] += reward
		# update reward in reward list
		# rewards_ls[action] = rewards_ls[action] + reward
		# update action value in Q estimated at time t action a
		# q_est_t_a[action] += (rewards_ls[action]- q_est_t_a[action])/num_actions[action]
		q_est_t_a[action] += (reward- q_est_t_a[action])/num_actions[action]
		if (action==mean_opt_action):
			opt_action[i] +=1
	return average_reward, opt_action

# temperatures = [0.01, 0.1, 1.0, 10.0, 100.0]
# avg_reward_ls = []
# opt_action_ls = []
# for i in range(len(temperatures)):
# 	avg_reward_ls.append([0]*1000)
# 	opt_action_ls.append([0]*1000)
# num_times = 2000
# for i in range(num_times):
# 	if(i%200==0):
# 		print(i)
# 	y_values = []	
# 	mean_dist = np.random.normal(0, 1, k)
# 	for t in range(len(temperatures)):
# 		avg_reward_ls[t], opt_action_ls[t] = softmax(temperatures[t],avg_reward_ls[t], opt_action_ls[t],mean_dist,k=10)
# for j in range(len(temperatures)):
# 	avg_reward_ls[j] = [x / num_times for x in avg_reward_ls[j]]
# 	plt.plot(x_values, avg_reward_ls[j], label = "temperature ="+str(temperatures[j])) 
# plt.xlabel('Steps') 
# plt.ylabel('Average reward') 
# plt.legend() 
# plt.show() 
# plt.close()

# for j in range(len(temperatures)):
# 	opt_action_ls[j] = [(x*100) / num_times for x in opt_action_ls[j]]
# 	plt.plot(x_values, opt_action_ls[j], label = "temperature ="+str(temperatures[j])) 
# 	plt.ylim(0.0, 100.0)
# plt.xlabel('Steps') 
# plt.ylabel('%  optimal action') 
# plt.legend() 
# plt.show() 
# plt.close()

################################################################################################################
################################################################################################################

# UCB-1
average_reward = []
avg_r = 0
def action_ucb(qt,t, nt, c=2,k=10):
	at = [0]*k
	for i in range(k):
		if(nt[i]==0):
			return i
		at[i] = qt[i] + (c*np.sqrt( np.log(t)/nt[i])) 
	return np.argmax(at,axis=0)

def ucb(average_reward, opt_action,mean_dist,k=10):
	q_est_t_a = [0]*k
	num_actions = [0]*k
	rewards_ls = [0]*k
	mean_opt_action = np.argmax(mean_dist)
	# print("mean_opt_action",mean_opt_action)
	avg_r = 0
	for i in range(1000):
		action = action_ucb(q_est_t_a, i+1, num_actions)
		# print("num_actions",num_actions)
		# print("action",action,mean_opt_action)
		num_actions[action] = num_actions[action] + 1
		sigma = 1
		mu = mean_dist[action]
		reward = float(np.random.normal(mu, sigma, 1))
		average_reward[i] += reward
		# rewards_ls[action] = rewards_ls[action] + reward
		# update action value in Q estimated at time t action a
		q_est_t_a[action] += (reward- q_est_t_a[action])/num_actions[action]
		# q_est_t_a[action] += (rewards_ls[action]- q_est_t_a[action])/num_actions[action]
		# print(q_est_t_a)
		if (action==mean_opt_action):
				opt_action[i] +=1
	return average_reward, opt_action

# num_times = 2000
# avg_rwd_ucb, avg_reward_eps, avg_reward_soft, ucb_ls, eps_ls, softmax_ls = [0]*1000, [0]*1000, [0]*1000, [0]*1000, [0]*1000, [0]*1000
# for i in range(num_times):
# 	mean_dist = np.random.normal(0, 1, k)
# 	avg_rwd_ucb, ucb_ls = ucb(avg_rwd_ucb,ucb_ls,mean_dist,k=10)
# 	avg_reward_eps, eps_ls = epsilon_greedy(0.01, avg_reward_eps, eps_ls,mean_dist,k=10)  
# 	avg_reward_soft, softmax_ls = softmax(0.1, avg_reward_soft, softmax_ls,mean_dist,k=10)
# avg_rwd_ucb = [x / num_times for x in avg_rwd_ucb]
# plt.plot(x_values, avg_rwd_ucb, label = "UCB-1, c=2 ") 
# avg_reward_eps = [x / num_times for x in avg_reward_eps]
# plt.plot(x_values, avg_reward_eps, label = "epsilons = 0.01") 
# avg_reward_soft = [x / num_times for x in avg_reward_soft]
# plt.plot(x_values, avg_reward_soft, label = "temperature = 0.1 ") 
# plt.xlabel('Steps') 
# plt.ylabel('Average reward') 
# plt.legend() 
# plt.savefig("ucb_eps_soft_r")
# plt.show() 
# plt.close()

# ucb_ls = [(x*100) / num_times for x in ucb_ls]
# plt.plot(x_values, ucb_ls, label = "UCB-1, c=2 ") 
# eps_ls = [(x*100) / num_times for x in eps_ls]
# plt.plot(x_values, eps_ls, label = "epsilons = 0.01") 
# softmax_ls = [(x*100) / num_times for x in softmax_ls]
# plt.plot(x_values, softmax_ls, label = "temperature = 0.1 ") 
# plt.ylim(0.0, 100.0)
# plt.xlabel('Steps') 
# plt.ylabel('%  optimal action') 
# plt.legend() 
# plt.savefig("ucb_eps_soft")
# plt.show() 
# plt.close()

################################################################################################################
################################################################################################################

def MEA(epsilon, delta, opt_action_r, opt_action_not_r,k=10):
	num_actions = set(np.arange(k))
	mean_dist = np.random.normal(0, 1, k)
	mean_opt_action = set([np.argmax(mean_dist)])
	epsilon_l = epsilon / 4
	delta_l = delta / 2
	total_samples = 0
	while (len(num_actions) != 1): 
		q_est_t_a = dict(zip(list(num_actions), np.repeat(0., len(num_actions))))
		num_samples = int(np.log(3 / delta_l) * 2 / (epsilon_l ** 2))
		for key in q_est_t_a:
			for _ in range(num_samples):
				q_est_t_a[key] += np.random.normal(mean_dist[key], 1) 
				total_samples += 1 
		median = np.median(list(q_est_t_a.values()))
		remove_actions =[]
		for key, value in q_est_t_a.items():
			if value < median:
				remove_actions.append(key) 
		num_actions -= set(remove_actions) 
		epsilon_l *= 3 / 4
		delta_l *= 1 / 2
	if (num_actions==mean_opt_action):
		opt_action_r +=1
	else:
		opt_action_not_r += 1 
	return total_samples, opt_action_r, opt_action_not_r

# num_times = 2000
# epsilon_col = [0.9, 0.8, 0.5, 0.2, 0.1, 0.9, 0.8, 0.5, 0.2, 0.1, 0.9, 0.8, 0.5, 0.2, 0.1 ]
# delta_col = [0.9, 0.9, 0.9, 0.9, 0.9, 0.5,0.5, 0.5,0.5, 0.5, 0.1, 0.1, 0.1, 0.1, 0.1]
# total_samples_col = []
# per_opt_arm_retained_col = []
# other_action_col = []
# for k in range(15):
# 	epsilon, delta = epsilon_col[k], delta_col[k]
# 	print(k,epsilon, delta)
# 	opt_action_r, opt_action_not_r = 0, 0
# 	for i in range(num_times):
# 		total_samples, opt_action_r, opt_action_not_r = MEA(epsilon, delta, opt_action_r, opt_action_not_r,k=10)
# 	total_samples_col.append(total_samples)
# 	per_opt_arm_retained_col.append((opt_action_r*100)/num_times)
# 	other_action_col.append(opt_action_not_r)

# df = pd.DataFrame(list(zip(epsilon_col, delta_col,total_samples_col,per_opt_arm_retained_col,other_action_col)), 
#                columns =['epsilon', 'delta','total samples', '%  optimum arm chosen', 'no. times non-optimal arm']) 
# print(df)

# print(total_samples_col)
# print(per_opt_arm_retained_col)

# # plotting the points  
# plt.plot(epsilon_col[:5], total_samples_col[:5], color='red', linestyle='dashed', linewidth = 3, 
#          marker='o', markerfacecolor='black', markersize=12, label="delta = 0.9")
# plt.plot(epsilon_col[5:10], total_samples_col[5:10], color='blue', linestyle='dashed', linewidth = 3, 
#          marker='o', markerfacecolor='black', markersize=12, label="delta = 0.5")
# plt.plot(epsilon_col[10:], total_samples_col[10:], color='green', linestyle='dashed', linewidth = 3, 
#          marker='o', markerfacecolor='black', markersize=12, label="delta = 0.1")
# # plt.plot( epsilon_col,total_samples_col) 
# plt.xlabel('epsilon') 
# plt.ylabel('num of total samples') 
# plt.legend()
# plt.show() 
# plt.close()

# plt.plot(epsilon_col[:5], per_opt_arm_retained_col[:5], color='red', linestyle='dashed', linewidth = 3, 
#          marker='o', markerfacecolor='black', markersize=12, label="delta = 0.9")
# plt.plot(epsilon_col[5:10], per_opt_arm_retained_col[5:10], color='blue', linestyle='dashed', linewidth = 3, 
#          marker='o', markerfacecolor='black', markersize=12, label="delta = 0.5")
# plt.plot(epsilon_col[10:], per_opt_arm_retained_col[10:], color='green', linestyle='dashed', linewidth = 3, 
#          marker='o', markerfacecolor='black', markersize=12, label="delta = 0.1")
# # plt.plot( epsilon_col,per_opt_arm_retained_col) 
# plt.xlabel('epsilon') 
# plt.ylabel('%  optimal action') 
# plt.legend()
# plt.show() 


num_times = 2000
epsilon_col = [0.9, 0.1, 0.9, 0.1, 0.9,  0.1 ]
delta_col = [0.9, 0.9, 0.5, 0.5,  0.1, 0.1]
total_samples_col = []
per_opt_arm_retained_col = []
other_action_col = []
for k in range(6):
	epsilon, delta = epsilon_col[k], delta_col[k]
	print(k,epsilon, delta)
	opt_action_r, opt_action_not_r = 0, 0
	for i in range(num_times):
		total_samples, opt_action_r, opt_action_not_r = MEA(epsilon, delta, opt_action_r, opt_action_not_r,k=1000)
	total_samples_col.append(total_samples)
	per_opt_arm_retained_col.append((opt_action_r*100)/num_times)
	other_action_col.append(opt_action_not_r)

df = pd.DataFrame(list(zip(epsilon_col, delta_col,total_samples_col,per_opt_arm_retained_col,other_action_col)), 
               columns =['epsilon', 'delta','total samples', '%  optimum arm chosen', 'no. times non-optimal arm']) 
print(df)

print(total_samples_col)
print(per_opt_arm_retained_col)

# plotting the points  
plt.plot(epsilon_col[:2], total_samples_col[:2], color='red', linestyle='dashed', linewidth = 3, 
         marker='o', markerfacecolor='black', markersize=12, label="delta = 0.9")
plt.plot(epsilon_col[2:4], total_samples_col[2:4], color='blue', linestyle='dashed', linewidth = 3, 
         marker='o', markerfacecolor='black', markersize=12, label="delta = 0.5")
plt.plot(epsilon_col[4:], total_samples_col[4:], color='green', linestyle='dashed', linewidth = 3, 
         marker='o', markerfacecolor='black', markersize=12, label="delta = 0.1")
# plt.plot( epsilon_col,total_samples_col) 
plt.xlabel('epsilon') 
plt.ylabel('num of total samples') 
plt.legend()
plt.show() 
plt.close()

plt.plot(epsilon_col[:2], per_opt_arm_retained_col[:2], color='red', linestyle='dashed', linewidth = 3, 
         marker='o', markerfacecolor='black', markersize=12, label="delta = 0.9")
plt.plot(epsilon_col[2:4], per_opt_arm_retained_col[2:4], color='blue', linestyle='dashed', linewidth = 3, 
         marker='o', markerfacecolor='black', markersize=12, label="delta = 0.5")
plt.plot(epsilon_col[4:], per_opt_arm_retained_col[4:], color='green', linestyle='dashed', linewidth = 3, 
         marker='o', markerfacecolor='black', markersize=12, label="delta = 0.1")
# plt.plot( epsilon_col,per_opt_arm_retained_col) 
plt.xlabel('epsilon') 
plt.ylabel('%  optimal action') 
plt.legend()
plt.show() 
#!/usr/bin/env python
import click
import gym
import numpy as np 
import pandas as pd 
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt 

gym.envs.register(
     id='chakra-v0',
     entry_point='rlpa2:chakra',
     # max_episode_steps=150,
     # kwargs={'size' : 1, 'init_state' : 10., 'state_bound' : np.inf},
)

def include_bias(obs):
    return np.append(obs,1)

def chakra_get_action(theta, ob, rng=np.random):
    ob_1 = include_bias(ob)
    mean = theta.dot(ob_1)
    return rng.normal(loc=mean, scale=1.)        

def log_pi_a_s(ob, theta):
    sdash = include_bias(ob)
    theta_transpose = np.transpose(theta)
    mean = sdash.dot(theta_transpose)
    cov = [[1, 0], [0, 1]]
    policy = np.squeeze(np.random.multivariate_normal(mean, cov, 1))
    return policy

def grad_log_pi_a_s(policy, ob):
    sdash = include_bias(ob)
    sdash_d = np.array(sdash)[np.newaxis]
    policy_d = np.array(policy)[np.newaxis]
    grad_pi = np.transpose(policy_d).dot(sdash_d)
    return grad_pi

def discount_rewards(rewards, gamma):
    discounted_rewards = np.zeros(len(rewards))
    cumulative_rewards = 0
    for i in reversed(range(0, len(rewards))):
        cumulative_rewards = cumulative_rewards * gamma + rewards[i]
        discounted_rewards[i] = cumulative_rewards
    return discounted_rewards

def run_episode(env,theta, ob, rng, get_action, batch_size = 1, T= 1000, tdash=5, gamma=0.9 ):
    ob = env.reset()
    episode_reward = 0.0
    done = False
    traj_sz = []
    trajectory_reward = []
    for n_sample in range(batch_size):
        batch_reward = 0.0
        # ob = start_ob
        states_ls, actions_ls, rewards_ls, transitions_ls,grad_sum = [], [], [], [], []
        for t in range(T+1):
            # print("ob",ob)
            states_ls.append(ob)
            action = get_action(theta, ob, rng)
            # print("action",action)
            actions_ls.append(action)
            next_ob, reward, done, _ = env.step(action)
            rewards_ls.append(reward)
            # print("next_ob",next_ob)
            # print("reward, done", reward,done)
            env.render("human")
            transitions_ls.append((ob, action, reward))
            if done:
                traj_sz.append(t)
                print("here done", ob)
                t = T + 1
            if (t==T):
                traj_sz.append(t)
            batch_reward += reward
            ob = next_ob
        discount_reward_ls = [] 
        gamma_pow = 0
        discounts = [gamma**i for i in range(len(rewards_ls)+1)]
        for idx, transition in enumerate(transitions_ls):
            st, act, rew = transition
            # gamma_pow =  gamma_pow + np.power(gamma, tdash - idx)
            # discount_reward = gamma_pow*rew
            discount_reward = rewards_ls[idx]*discounts[idx]
            discount_reward_ls.append(discount_reward)
            pol = log_pi_a_s(st, theta)
            grad_pol = grad_log_pi_a_s(pol, st)
            if idx == 0:
                grad_pol_f = grad_pol*discount_reward
            else:
                grad_pol_f += grad_pol*discount_reward
        trajectory_reward.append(np.sum(discount_reward_ls))
    return  sum(traj_sz)/batch_size, np.sum(trajectory_reward)/batch_size, grad_pol_f/batch_size


learning_rate = 0.5
batch_size = 2
max_itr = 200000
T = 100 # trajectory length
gamma = 0.9
tdash = 20


@click.command()
@click.argument("env_id", type=str, default="chakra")
def main(env_id):
    # Register the environment
    rng = np.random.RandomState(42)
    learning_rate = 0.5

    if env_id == 'chakra':
        # from rlpa2 import chakra
        env = gym.make('chakra-v0')
        get_action = chakra_get_action
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
    else:
        raise ValueError(
            "Unsupported environment: must be 'chakra' ")

    env.seed(42)
    while True:
        ob = env.reset()
        done = False
        # Only render the first trajectory
        # Collect a new trajectory
        rewards,episodes, traj_mn = [], [], []
        for itr in range(max_itr):
            if(itr>100):
                learning_rate = 0.01
            traj_bt, avg_reward, grad_traj = run_episode(env,theta, ob, rng, get_action, batch_size, T, tdash, gamma )
            a = np.squeeze(np.array(grad_traj))
            print("avg_reward",avg_reward)
            theta += learning_rate*a
            if( itr%50==0):
                print("episode, average_rew, theta , traj_bt: ", itr+1, avg_reward, theta, traj_bt)
            # action = get_action(theta, ob, rng=rng)
            # next_ob, rew, done, _ = env.step(action)
            # ob = next_ob
            # env.render()
            rewards.append(avg_reward)
            episodes.append(itr+1)
            traj_mn.append(traj_bt)
            # print("Episode reward: %.2f" % np.sum(rewards))
            if( itr%5000==0):
                print(itr)
                df = pd.DataFrame(list(zip(episodes, rewards, traj_mn)), columns =["episodes", "rewards", "traj_mn"]) 
                # print("episodes",episodes)
                df.to_csv("b2"+str(itr)+".csv")
                # print("rewards",rewards)
                # print("traj_mn",traj_mn)
                plt.plot(episodes, rewards, label = "gamma = " + str(gamma) + " batch_size = "+ str(batch_size)+" learning_rate = " + str(learning_rate)) 
                plt.xlabel('number of episodes') # X axis label
                plt.ylabel('reward per episode') # Y axis label
                # plt.yticks(np.arange(min(avg_rewards), max(avg_rewards)+10, 10.0))
                plt.title("Policy Gradient rewards " ) # Show legend
                # plt.show() # show the plot
                plt.legend() # Show legend
                plt.savefig("plts/"+"pg_"+str(gamma)+str(batch_size)+ str(learning_rate)+"rew.png")
                plt.close() # Close the plot
                # plt.plot(episode_ls, steps_ls) 
                plt.plot(episodes, traj_mn, label = "gamma = " + str(gamma)+" batch_size = "+ str(batch_size)+" learning_rate = " + str(learning_rate)) 
                plt.xlabel('number of episodes') # X axis label
                plt.ylabel('steps per episode') # Y axis label
                # plt.yticks(np.arange(min(avg_steps)-10, max(avg_steps)+10, 10.0))
                plt.title("Policy Gradient steps " ) # Show legend
                # plt.show() # show the plot
                plt.legend() # Show legend
                plt.savefig("plts/"+"pg_"+str(gamma)+str(batch_size)+ str(learning_rate)+"step.png")
                plt.close() # Close the plot



if __name__ == "__main__":
    main()

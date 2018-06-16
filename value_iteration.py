""" ---------------- Value iteration implementation------------- 
 Assumptions:
 Discrete, Tractable State & Action space
 Environment dynamics (Transition function & Reward function) are known
"""
import numpy as np
import gym
from gym import wrappers


""" Let's define the Value iteration algorithm first. 
    Value Iteration:
	1. Value function evaluation
	2. Value function improvement

 Value function evaluation:
  	For iteration in range(limit):
	For all states s in env state_space:
		For all actions a in state s:
			q_value[s,a] = reward + sum( state_value[s_] * p for all new states s_ with transition probabilities p )
		state_value[s] = max(q_value[s,a] for all actions a)

"""

def value_iteration(env, gamma=0.99, eps= 1e-27, iterations = 10000000):
	"""
			Value iteration function that fits the optimal value function
	"""

	state_value = np.zeros(env.observation_space.n,dtype=np.double)
	print ("\n Starting Value Iteration....")
	for i in range(iterations):
		prev_state_value= np.copy(state_value,dtype=np.double)
		for s in range(env.observation_space.n):
			q_sa = [sum(prob * (reward + gamma * prev_state_value[s_]) for prob,s_,reward,_ in env.env.P[s][a]) for a in range(env.action_space.n)]
			state_value[s] = max(q_sa)

		error = np.sum(np.fabs(state_value-prev_state_value))
		if (error < eps):
			print ("\nValue Iteration has converged after " + str(i) + " iterations.")
			break
		else: print ("\n Iteration No:" + str(i) + " Value error: " + str(error))
	return state_value

def greedy_policy(current_state, env, state_value, gamma=0.99):
	"""
	Greedy policy wrt to the state-action value

	:return: action to be taken in current state
	"""
	q_sa = [sum(prob * (reward + gamma * state_value[new_state]) for prob, new_state, reward, _ in env.env.P[current_state][a]) for a in
			range(env.action_space.n)]
	return np.argmax(q_sa)

def evaluate_policy(state_value, n_episodes, env, render= True, gamma=0.99):
	"""
	Evaluates policy on 'n_episodes' epsiodes of the environment
	:return: average reward
	"""
	total_reward = 0
	for i in range(n_episodes):
		gain = 0
		obs = env.reset()
		step_no = 0
		while True:
			if render:
				env.render()
			obs, reward, done, _ = env.step(int(greedy_policy(obs,env,state_value,gamma)))

			gain += (gamma**step_no * reward)
			step_no += 1
			if done:
				print("\nGain in episode: " + str(i) + "was: "+ str(gain))
				total_reward += gain
				break
	return np.double(np.double(total_reward)/n_episodes)



if __name__ == '__main__':

	env_name = 'FrozenLake8x8-v0'
	n_episodes = 10000
	env= gym.make(env_name)

	optimal_state_value= value_iteration(env=env)
	print ("Average reward for current policy across " + str(n_episodes) + " episodes is: "+ str(evaluate_policy(state_value=optimal_state_value,n_episodes=n_episodes,env=env,render=True)))




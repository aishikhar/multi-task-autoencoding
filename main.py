import argparse
import numpy as np
import gym
from keras.optimizers import Adam
from dqn import DQN_atari
from rl.callbacks import FileLogger, ModelIntervalCheckpoint

NB_STEPS_PER_TSK= 5000
NB_TASKS= 50
CALLBACK_INTERVAL= 5000
LOG_INTERVAL= 4990

parser = argparse.ArgumentParser()
parser.add_argument('--mode', choices=['train', 'test'], default='train')
parser.add_argument('--env-name', type=str, default='BreakoutDeterministic-v4')
parser.add_argument('--weights', type=str, default=None)
args = parser.parse_args()

env = gym.make(args.env_name)
np.random.seed(123)
env.seed(123)
nb_actions = env.action_space.n

# Initialize the Model-free DQN
dqn = DQN_atari(nb_actions)
dqn.compile(Adam(lr=.00025), metrics=['mae'])

if args.mode == 'train':

    weights_filename = 'dqn_{}_weights.h5f'.format(args.env_name)
    checkpoint_weights_filename = 'dqn_mem_' + args.env_name + '_weights_{step}.h5f'
    log_filename = 'dqn_mem_{}_log.json'.format(args.env_name)
    callbacks = [ModelIntervalCheckpoint(checkpoint_weights_filename, interval=CALLBACK_INTERVAL)]
    callbacks += [FileLogger(log_filename, interval=100)]
    dqn.fit(env, callbacks=callbacks, nb_steps=NB_STEPS_PER_TSK, log_interval=LOG_INTERVAL,visualize=True)

    # Save the weights of the Agent
    dqn.save_weights(weights_filename, overwrite=True)

    # Finally, evaluate our algorithm for 10 episodes.
    dqn.test(env, nb_episodes=20, visualize=True)
elif args.mode == 'test':
    weights_filename = 'dqn_mem_{}_weights.h5f'.format(args.env_name)
    if args.weights:
        weights_filename = args.weights
    dqn.load_weights(weights_filename)
    dqn.test(env, nb_episodes=10, visualize=False)

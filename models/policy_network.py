""" 

Q-value network based on the original DQN paper (Mnih et. al. 2013).


"""
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Convolution2D, Permute
import keras.backend as K

def Q_network(input_shape,nb_actions):

	model = Sequential()

	if K.image_dim_ordering() == 'tf':
	    # (width, height, channels)
	    model.add(Permute((2, 3, 1), input_shape=input_shape))
	elif K.image_dim_ordering() == 'th':
	    # (channels, width, height)
	    model.add(Permute((1, 2, 3), input_shape=input_shape))
	else:
	    raise RuntimeError('Unknown image_dim_ordering.')
	
	model.add(Convolution2D(32, 8, 8, subsample=(4, 4)))
	model.add(Activation('relu'))
	model.add(Convolution2D(64, 4, 4, subsample=(2, 2)))
	model.add(Activation('relu'))
	model.add(Convolution2D(64, 3, 3, subsample=(1, 1)))
	model.add(Activation('relu'))
	model.add(Flatten())
	model.add(Dense(512))
	model.add(Activation('relu'))
	model.add(Dense(nb_actions))
	model.add(Activation('linear'))
	print(model.summary())
	return model 
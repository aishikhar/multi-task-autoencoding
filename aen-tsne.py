

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
from tensorflow.examples.tutorials.mnist import input_data

import numpy as np

import argparse
import sys
import math
import os


FLAGS = None
NB_TEST_DATA = 10000


def autoencoder(dimensions=[784, 512, 256, 64]):
    """Build a deep autoencoder w/ tied weights.
    Parameters
    ----------
    dimensions : list, optional
        The number of neurons for each layer of the autoencoder.
    Returns
    -------
    x : Tensor
        Input placeholder to the network
    z : Tensor
        Inner-most latent representation
    y : Tensor
        Output reconstruction of the input
    cost : Tensor
        Overall cost to use for training
    """
    # %% input to the network
    x = tf.placeholder(tf.float32, [None, dimensions[0]], name='x')
    current_input = x

    # %% Build the encoder
    encoder = []
    for layer_i, n_output in enumerate(dimensions[1:]):
        n_input = int(current_input.get_shape()[1])
        W = tf.Variable(
            tf.random_uniform([n_input, n_output],
                              -1.0 / math.sqrt(n_input),
                              1.0 / math.sqrt(n_input)))
        b = tf.Variable(tf.zeros([n_output]))
        encoder.append(W)
        output = tf.nn.tanh(tf.matmul(current_input, W) + b)
        current_input = output

    # Latent representation (embedding, neural coding)
    z = current_input
    encoder.reverse()
    # Build the right part of the code for t-sne representation of te


    # Build the decoder using the same weights
    for layer_i, n_output in enumerate(dimensions[:-1][::-1]):
        W = tf.transpose(encoder[layer_i])
        b = tf.Variable(tf.zeros([n_output]))
        output = tf.nn.tanh(tf.matmul(current_input, W) + b)
        current_input = output

    # Now have the reconstruction through the network
    y = current_input

    # Cost function measures pixel-wise difference
    cost = tf.reduce_sum(tf.square(y - x))
    return {'x': x, 'z': z, 'y': y, 'cost': cost}


def train_autoencoder_and_embed():
    """Test the autoencoder using MNIST."""
    import tensorflow as tf
    import tensorflow.examples.tutorials.mnist.input_data as input_data
    import matplotlib.pyplot as plt 

    # load MNIST as before
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    mean_img = np.mean(mnist.train.images, axis=0)
    ae = autoencoder(dimensions=[784, 256, 64])

    learning_rate = 0.001
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(ae['cost'])

    # We create a session to use the graph
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # Fit all training data
    batch_size = 50
    n_epochs = 30
    for epoch_i in range(n_epochs):
        for batch_i in range(mnist.train.num_examples // batch_size):
            batch_xs, _ = mnist.train.next_batch(batch_size)
            train = np.array([img - mean_img for img in batch_xs])
            sess.run(optimizer, feed_dict={ae['x']: train})
        print(epoch_i, sess.run(ae['cost'], feed_dict={ae['x']: train}))

    # Get embeddings.
    # If you have too much to get and that it does not fit in memory, you may
    # need to use a batch size or to force to use the CPU rather than the GPU.
    test = [img - mean_img for img in mnist.test.images]
    embedded_data = sess.run(
        ae['z'],
        feed_dict={ae['x']: test}
    )
    return embedded_data, sess


def generate_embeddings():
    # Load data, train an autoencoder and transform data
    embedded_data, sess = train_autoencoder_and_embed()

    # Input set for Embedded TensorBoard visualization
    # Performed with cpu to conserve memory and processing power
    with tf.device("/cpu:0"):
        embedding = tf.Variable(tf.stack(embedded_data, axis=0), trainable=False, name='embedding')

    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver()
    writer = tf.summary.FileWriter(FLAGS.log_dir + '/projector', sess.graph)

    # Add embedding tensorboard visualization. Need tensorflow version
    # >= 0.12.0RC0
    config = projector.ProjectorConfig()
    embed= config.embeddings.add()
    embed.tensor_name = 'embedding:0'
    embed.metadata_path = os.path.join(FLAGS.log_dir + '/projector/metadata.tsv')
    embed.sprite.image_path = os.path.join(FLAGS.data_dir + '/mnist_10k_sprite.png')

    # Specify the width and height of a single thumbnail.
    embed.sprite.single_image_dim.extend([28, 28])
    projector.visualize_embeddings(writer, config)

    # We save the embeddings for TensorBoard, setting the global step as
    # The number of data examples
    saver.save(sess, os.path.join(
        FLAGS.log_dir, 'projector/a_model.ckpt'), global_step=NB_TEST_DATA)

    sess.close()

def generate_metadata_file():
    # Import data
    mnist = input_data.read_data_sets(FLAGS.data_dir,
                                      one_hot=True)
    # The ".tsv" file will contain one number per row to point to the good label
    # for each test example in the dataset.
    # For example, labels could be saved as plain text on those lines if needed.
    # In our case we have only 10 possible different labels, so their
    # "uniqueness" is recognised to later associate colors automatically in
    # TensorBoard.
    def save_metadata(file):
        with open(file, 'w') as f:
            for i in range(NB_TEST_DATA):
                c = np.nonzero(mnist.test.labels[::1])[1:][0][i]
                f.write('{}\n'.format(c))

    save_metadata(FLAGS.log_dir + '/projector/metadata.tsv')

def main(_):
    if tf.gfile.Exists(FLAGS.log_dir + '/projector'):
        tf.gfile.DeleteRecursively(FLAGS.log_dir + '/projector')
        tf.gfile.MkDir(FLAGS.log_dir + '/projector')
    tf.gfile.MakeDirs(FLAGS.log_dir  + '/projector') # fix the directory to be created
    generate_metadata_file()
    generate_embeddings()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./mnist_data',
                        help='Directory for storing input data')
    parser.add_argument('--log_dir', type=str, default='./logs',
                        help='Summaries log directory')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
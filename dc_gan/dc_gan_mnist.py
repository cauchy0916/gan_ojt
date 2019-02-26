"""
	# Author: Sungbum Park (spark0916@ncsoft.com)
	# Last Modified: Feb. 25, 2019
	# Version: v0.1
	# Comment: 
		- DC GAN implementation using TensorFlow for MNIST
	# Reference:
		- https://github.com/FelixMohr/Deep-learning-with-Python/blob/master/DCGAN-MNIST.ipynb
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os


# Output path
if not os.path.exists('out/'):
	os.makedirs('out/')  


# Image plot
def plot(samples):
	fig = plt.figure(figsize=(4, 4))
	gs = gridspec.GridSpec(4, 4)
	gs.update(wspace=0.05, hspace=0.05)

	for i, sample in enumerate(samples):
		ax = plt.subplot(gs[i])
		plt.axis('off')
		ax.set_xticklabels([])
		ax.set_yticklabels([])
		ax.set_aspect('equal')
		plt.imshow(sample.reshape(28, 28), cmap='Greys_r')

	return fig

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('../../MNIST_data', one_hot=True)

# Hyper parameters
batch_size = 64
z_dim = 64
n_epoch = 3
n_batch = int(mnist.train.images.shape[0] / batch_size)
keep_prob_train = 0.6


X = tf.placeholder(dtype=tf.float32, shape=[None, 28, 28], name='X')
Z = tf.placeholder(dtype=tf.float32, shape=[None, z_dim])
keep_prob = tf.placeholder(dtype=tf.float32, name='keep_prob')
is_training = tf.placeholder(dtype=tf.bool, name='is_training')


# Leaky ReLU
def lrelu(x):
	return tf.maximum(x, tf.multiply(x, 0.2))


# Binary Cross Entropy
def binary_cross_entropy(x, z):
	eps = 1e-12
	return (-(x * tf.log(z + eps) + (1. - x) * tf.log(1. - z + eps)))


# Discriminator
def discriminator(img_in, reuse=None, keep_prob=keep_prob):
	activation = lrelu
	with tf.variable_scope("discriminator", reuse=reuse):            
		x = tf.reshape(img_in, shape=[-1, 28, 28, 1])

		# (28 x 28 x 1) > (14 x 14 x 64)
		x = tf.layers.conv2d(x, kernel_size=5, filters=64, strides=2, padding='same', activation=activation)
		x = tf.layers.dropout(x, keep_prob)

		# (14 x 14 x 16) > (14 x 14 x 64)
		x = tf.layers.conv2d(x, kernel_size=5, filters=64, strides=1, padding='same', activation=activation)
		x = tf.layers.dropout(x, keep_prob)

		# (14 x 14 x 16) > (14 x 14 x 64)
		x = tf.layers.conv2d(x, kernel_size=5, filters=64, strides=1, padding='same', activation=activation)
		x = tf.layers.dropout(x, keep_prob)

		# (14 x 28 x 16) > (1 x 12,544)
		x = tf.contrib.layers.flatten(x)

		# (12,544) > (128)
		x = tf.layers.dense(x, units=128, activation=activation)

		# (128) > 1
		x = tf.layers.dense(x, units=1, activation=tf.nn.sigmoid)
		return x


# Generator
def generator(z, keep_prob=keep_prob, is_training=is_training):
	activation = lrelu
	momentum = 0.99
	with tf.variable_scope("generator", reuse=None):
		d1 = 4
		d2 = 1

		# (64) > (16)
		x = tf.layers.dense(z, units=d1 * d1 * d2, activation=activation)
		x = tf.layers.dropout(x, keep_prob)      
		x = tf.contrib.layers.batch_norm(x, is_training=is_training, decay=momentum)  

		# (16) > (4 x 4 x 1)
		x = tf.reshape(x, shape=[-1, d1, d1, d2])

		# (4 x 4 x 1) > (7 x 7 x 1)
		x = tf.image.resize_images(x, size=[7, 7])

		# (7 x 7 x 1) > (14 x 14 x 64)
		x = tf.layers.conv2d_transpose(x, kernel_size=5, filters=64, strides=2, padding='same', activation=activation)
		x = tf.layers.dropout(x, keep_prob)       
		x = tf.contrib.layers.batch_norm(x, is_training=is_training, decay=momentum)

		# (14 x 14 x 64) > (28 x 28 x 64) 
		x = tf.layers.conv2d_transpose(x, kernel_size=5, filters=64, strides=2, padding='same', activation=activation)
		x = tf.layers.dropout(x, keep_prob)
		x = tf.contrib.layers.batch_norm(x, is_training=is_training, decay=momentum)

		# (28 x 28 x 64) > (28 x 28 x 64)         
		x = tf.layers.conv2d_transpose(x, kernel_size=5, filters=64, strides=1, padding='same', activation=activation)
		x = tf.layers.dropout(x, keep_prob)
		x = tf.contrib.layers.batch_norm(x, is_training=is_training, decay=momentum)

		# (28 x 28 x 64) > (28 x 28 x 1)         
		x = tf.layers.conv2d_transpose(x, kernel_size=5, filters=1, strides=1, padding='same', activation=tf.nn.sigmoid)
		return x


# Uniform prior generation for G(Z)
def sample_Z(m, n):
	return np.random.uniform(-1., 1., size=[m, n])


# Losses and optimizers
g = generator(Z, keep_prob, is_training)
d_real = discriminator(X)
d_fake = discriminator(g, reuse=True)

vars_g = [var for var in tf.trainable_variables() if var.name.startswith("generator")]
vars_d = [var for var in tf.trainable_variables() if var.name.startswith("discriminator")]


d_reg = tf.contrib.layers.apply_regularization(tf.contrib.layers.l2_regularizer(1e-6), vars_d)
g_reg = tf.contrib.layers.apply_regularization(tf.contrib.layers.l2_regularizer(1e-6), vars_g)


loss_d_real = binary_cross_entropy(tf.ones_like(d_real), d_real)
loss_d_fake = binary_cross_entropy(tf.zeros_like(d_fake), d_fake)
loss_g = tf.reduce_mean(binary_cross_entropy(tf.ones_like(d_fake), d_fake))
loss_d = tf.reduce_mean(0.5 * (loss_d_real + loss_d_fake))
# loss_d_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_real, labels=tf.ones_like(d_real)))
# loss_d_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_fake, labels=tf.zeros_like(d_fake)))
# loss_d = 0.5 * (loss_d_real + loss_d_fake)
# loss_g = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_fake, labels=tf.ones_like(d_fake)))


update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
	optimizer_d = tf.train.RMSPropOptimizer(learning_rate=0.00015).minimize(loss_d + d_reg, var_list=vars_d)
	optimizer_g = tf.train.RMSPropOptimizer(learning_rate=0.00015).minimize(loss_g + g_reg, var_list=vars_g)
  
	
sess = tf.Session()
sess.run(tf.global_variables_initializer())


# Learning curve
hist = 0
history = {'val_D_loss': [], 'val_G_loss': []}


# Training & Generation Result Dump
for i in range(n_epoch):
	# MNIST image generation using the current network
	samples = sess.run(g, feed_dict = {Z: sample_Z(16, z_dim), keep_prob: 1.0, is_training:False})
	fig = plot(samples)
	plt.savefig('out/{}.png'.format(str(i).zfill(3)), bbox_inches='tight')
	plt.close()


	# Training
	for itr in range(n_batch):
		hist += 1

		train_d = True
		train_g = True
	
		n = sample_Z(batch_size, z_dim)
#		batch = [np.reshape(b, [28, 28]) for b in mnist.train.next_batch(batch_size=batch_size)[0]]
		batch, _ = mnist.train.next_batch(batch_size)
		batch = [np.reshape(64, [28, 28])]
#		batch = tf.reshape(X_mb, shape=[-1, 28, 28, 1])        

		d_real_ls, d_fake_ls, g_ls, d_ls = sess.run([loss_d_real, loss_d_fake, loss_g, loss_d], feed_dict={X: batch, Z: n, keep_prob: keep_prob_train, is_training:True})
	
		d_real_ls = np.mean(d_real_ls)
		d_fake_ls = np.mean(d_fake_ls)
	
		if g_ls * 1.5 < d_ls:
			train_g = False
			pass
		if d_ls * 2 < g_ls:
			train_d = False
			pass
	
		if train_d:
			sess.run(optimizer_d, feed_dict={Z: n, X: batch, keep_prob: keep_prob_train, is_training:True})
		if train_g:
			sess.run(optimizer_g, feed_dict={Z: n, keep_prob: keep_prob_train, is_training:True})

		history['val_D_loss'].append(d_ls)
		history['val_G_loss'].append(g_ls)


	# Loss update
	print('Epoch: {}	D Loss: {:.4}	G Loss:{}'.format(i, d_ls, g_ls))


# Graph plot
fig_graph = plt.figure()

sub_D_loss = fig_graph.add_subplot(211)
sub_D_loss.plot(range(hist), history['val_D_loss'], color='red')
sub_D_loss.set_xlabel('Discriminator Loss')

sub_G_loss = fig_graph.add_subplot(212)
sub_G_loss.plot(range(hist), history['val_G_loss'], color='blue')
sub_G_loss.set_xlabel('Generator Loss')

sub_D_loss.grid(linestyle='--', color='lavender')
sub_G_loss.grid(linestyle='--', color='lavender')

plt.savefig('out/dc_gan_loss.png')
plt.close(fig_graph)
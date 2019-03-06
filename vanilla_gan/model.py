"""
	# Author: Sungbum Park	
	# Version: 0.1
	# Last Modified: Mar. 5, 2019
	# Comment
	   : Conditional GAN for MNIST - model
"""
from __future__ import division
import os
import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np


from module import *
from utils import *


class cgan_mnist(object):
	def __init__(self, sess, args):
		self.mnist = input_data.read_data_sets(args.dataset_dir, one_hot=True)

		self.img_dim = self.mnist.train.images.shape[1]
		self.y_dim = self.mnist.train.labels.shape[1]
		self.z_dim = args.z_dim

		self.sess = sess
		self.batch_size = args.batch_size
		self.n_batch = int(self.mnist.train.images.shape[0] / self.batch_size)

		self.discriminator = discriminator
		self.generator = generator

		self._build_model()
#		self.saver = tf.train.Saver()

	def _build_model(self):
		""" Input, outputs """
		self.x = tf.placeholder(tf.float32, [None, self.img_dim])
		self.y = tf.placeholder(tf.float32, [None, self.y_dim])
		self.z = tf.placeholder(tf.float32, [None, self.z_dim])

		""" Generator, discriminator """
		self.g_sample = self.generator(self.z, self.y, self.img_dim)
		self.d_logit_real = self.discriminator(self.x, self.y)
		self.d_logit_fake = self.discriminator(self.g_sample, self.y, reuse=True)

		""" Loss function """
		self.d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.d_logit_real,
																				  labels=tf.ones_like(self.d_logit_real)))
		self.d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.d_logit_fake,
																				  labels=tf.zeros_like(self.d_logit_fake)))
		self.d_loss = self.d_loss_real + self.d_loss_fake
		self.g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.d_logit_fake,
																			 labels=tf.ones_like(self.d_logit_fake)))

		""" Summary """
		self.d_loss_sum = tf.summary.scalar("d_loss", self.d_loss)
		self.d_loss_real_sum = tf.summary.scalar("d_loss_real", self.d_loss_real)
		self.d_loss_fake_sum = tf.summary.scalar("d_loss_fake", self.d_loss_fake)

		self.g_loss_sum = tf.summary.scalar("g_loss", self.g_loss)

		self.d_sum = tf.summary.merge([self.d_loss_sum, self.d_loss_real_sum, self.d_loss_fake_sum])
		self.g_sum = tf.summary.merge([self.g_loss_sum])


		""" Variables """
		t_vars = tf.trainable_variables()
		self.d_vars = [var for var in t_vars if var.name.startswith("discriminator")]
		self.g_vars = [var for var in t_vars if var.name.startswith("generator")]
		for var in t_vars: print(var.name)


	def train(self, args):
		"""Train conditional gan"""
		self.lr = tf.placeholder(tf.float32, None, name='learning_rate')
		self.d_optimize = tf.train.AdamOptimizer(self.lr, beta1=args.beta1) \
			.minimize(self.d_loss, var_list=self.d_vars)
		self.g_optimize = tf.train.AdamOptimizer(self.lr, beta1=args.beta1) \
			.minimize(self.g_loss, var_list=self.g_vars)

		self.sess.run(tf.global_variables_initializer())
		self.writer = tf.summary.FileWriter("./logs", self.sess.graph)

		counter = 1
		start_time = time.time()

		"""
		if args.continue_train:
			if self.load(args.checkpoint_dir):
				print(" [*] Load SUCCESS")
			else:
				print(" [!] Load failed...")
		"""               

		for epoch in range(args.epoch):
			""" MNIST image geneartion """
			self.sample_model(args.sample_dir, epoch)

			lr = args.lr

			for idx in range(self.n_batch):
				x_mb, y_mb = self.mnist.train.next_batch(self.batch_size)

				_, summary_curr = self.sess.run([self.d_optimize, self.d_sum], 
									   feed_dict={self.x: x_mb, self.z: sample_z(self.batch_size, self.z_dim), self.y: y_mb, self.lr: lr})
				self.writer.add_summary(summary_curr, counter)

				_, summary_curr = self.sess.run([self.g_optimize, self.g_sum],
									   feed_dict={self.z: sample_z(self.batch_size, self.z_dim), self.y: y_mb, self.lr: lr})
				self.writer.add_summary(summary_curr, counter)

				counter += 1

			""" Loss print """
#			print(("Epoch: [%2d] D_loss: [%4.4f] G_loss: [%4.4f] Time: %4.4f" % (
#					epoch, self.d_sum[0], self.g_sum[0], time.time() - start_time)))
			print(("Epoch: [%2d] Time: %4.4f" % (epoch, time.time() - start_time)))



	def sample_model(self, sample_dir, epoch):
		y_sample = np.zeros(shape=[16,self.y_dim])
		for i in range(16):
			y_sample[i, i%10] = 1
		samples = self.sess.run(self.g_sample, feed_dict={self.z: sample_z(16, self.z_dim), self.y: y_sample})
		save_images(samples, './{}/{}.png'.format(sample_dir, epoch))

"""
	def save(self, checkpoint_dir, step):
		model_name = "cyclegan.model"
		model_dir = "%s_%s" % (self.dataset_dir, self.image_size)
		checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

		if not os.path.exists(checkpoint_dir):
			os.makedirs(checkpoint_dir)

		self.saver.save(self.sess,
						os.path.join(checkpoint_dir, model_name),
						global_step=step)

	def load(self, checkpoint_dir):
		print(" [*] Reading checkpoint...")

		model_dir = "%s_%s" % (self.dataset_dir, self.image_size)
		checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

		ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
		if ckpt and ckpt.model_checkpoint_path: 
			ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
			self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
			return True
		else:
			return False


		dataA = glob('./datasets/{}/*.*'.format(self.dataset_dir + '/testA'))
		dataB = glob('./datasets/{}/*.*'.format(self.dataset_dir + '/testB'))
		np.random.shuffle(dataA)
		np.random.shuffle(dataB)
		batch_files = list(zip(dataA[:self.batch_size], dataB[:self.batch_size]))
		sample_images = [load_train_data(batch_file, is_testing=True) for batch_file in batch_files]
		sample_images = np.array(sample_images).astype(np.float32)

		fake_A, fake_B = self.sess.run(
			[self.fake_A, self.fake_B],
			feed_dict={self.real_data: sample_images}
		)
		save_images(fake_A, [self.batch_size, 1],
					'./{}/A_{:02d}_{:04d}.jpg'.format(sample_dir, epoch, idx))
		save_images(fake_B, [self.batch_size, 1],
					'./{}/B_{:02d}_{:04d}.jpg'.format(sample_dir, epoch, idx))
"""
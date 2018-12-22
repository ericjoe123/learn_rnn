# -*- coding: UTF-8 -*-
import tensorflow as tf
import numpy as np
class model:
	def __init__(self):
		self.inputs = tf.placeholder(tf.float32, [None,4,1])
		self.output = self.encoder()
	def encoder(self):
		cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=10) # state_size = 128  
		outputs, h1 = tf.nn.dynamic_rnn(cell, self.inputs,dtype=tf.float32)
		return outputs
	def run(self):
		x=[[[1],[2],[3],[4]]]
		with tf.Session() as sess:
			sess.run(tf.global_variables_initializer())
			result=sess.run(self.output,feed_dict={self.inputs:x})
			print result


#encoder=model()
#encoder.run()
inputs = tf.placeholder(tf.float32, [None,4,1])
cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=10)
outputs, h1 = tf.nn.dynamic_rnn(cell, inputs,dtype=tf.float32)
x=[[[1],[2],[3],[4]]]
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	print sess.run([outputs,h1],feed_dict={inputs:x})
	


# -*- coding: UTF-8 -*-
import tensorflow as tf
import numpy as np
from tensorflow.contrib.seq2seq import *
from tensorflow.python.layers.core import Dense
class model():
	def __init__(self,mode):
		self.mode=mode
		self.inputs = tf.placeholder(tf.int32, [None,None])
		self.target = tf.placeholder(tf.int32, [None,None])
		self.embeddings_var = tf.get_variable("embedding_var", [5, 2])
		#self.embeddings_var = tf.Variable(tf.truncated_normal(shape=[5, 2], stddev=0.1),name='encoder_embedding')
		self.embedded_gno = tf.nn.embedding_lookup(self.embeddings_var,self.inputs )
		self.encoded_gno,self.h1 = self.encoder()
		self.predict_gno=self.decoder("predict")		
	def encoder(self):
		cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=10) 
		outputs, h1 = tf.nn.dynamic_rnn(cell, self.embedded_gno,dtype=tf.float32)
		return outputs,h1
	def decoder(self,mode):
		cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=10)
		#output_layer = tf.layers.Dense(5,kernel_initializer=tf.truncated_normal_initializer(mean=0.1,stddev=0.1))
		output_layer = tf.layers.Dense(2)
		max_len=tf.constant(4,dtype=tf.int32)
		if self.mode=='predict':
			start_tokens = tf.tile(tf.constant([0],dtype=tf.int32),[3],name='start_token')
			helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(self.embeddings_var,start_tokens,3)
			predicting_decoder = tf.contrib.seq2seq.BasicDecoder(cell,helper,self.h1,output_layer)
			predicting_decoder_output,_,_ = tf.contrib.seq2seq.dynamic_decode(predicting_decoder,impute_finished = True,
										maximum_iterations = max_len)
			return predicting_decoder_output 
	def train(self):
		x=[[2,1,2,3],[2,2,1,3],[2,2,1,3]]
		y=[[2,1,2,3],[2,2,1,3],[2,2,1,3]]
		with tf.Session() as sess:
			tf.get_variable_scope().reuse_variables()
			sess.run(tf.global_variables_initializer())
			result=sess.run([self.predict_gno,self.embeddings_var],feed_dict={self.inputs:x,self.target:y})
			print result


encoder=model("predict")
encoder.train()
'''
inputs = tf.placeholder(tf.float32, [None,4,1])
cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=10)
outputs, h1 = tf.nn.dynamic_rnn(cell, inputs,dtype=tf.float32)
x=[[[0],[1],[2],[3]]]
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	print sess.run([outputs,h1],feed_dict={inputs:x})
	
'''

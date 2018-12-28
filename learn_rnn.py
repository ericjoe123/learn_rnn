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
		self.inputs_len = tf.placeholder(tf.int32, [None,])
		self.target_len = tf.placeholder(tf.int32, [None,])
		self.embeddings_var = tf.get_variable("embedding_var", [5,3])
		#self.embeddings_var = tf.Variable(tf.truncated_normal(shape=[5, 2], stddev=0.1),name='encoder_embedding')
		self.embedded_gno = tf.nn.embedding_lookup(self.embeddings_var,self.inputs )
		self.embedded_target = tf.nn.embedding_lookup(self.embeddings_var,self.target)
		self.encoded_gno,self.h1 = self.encoder()
		self.predict_gno=self.decoder("predict")
		self.trained_gno=self.decoder("training")
		self.result = self.eva()
		
	def eva(self):
		training_logits = tf.identity(self.trained_gno.rnn_output,'logits')
		max_len=tf.constant(4,dtype=tf.int32)
		masks = tf.sequence_mask(self.target_len,max_len,dtype=tf.float32,name="masks")
		#cost = tf.contrib.seq2seq.sequence_loss(training_logits,self.target,masks)	
		return masks,self.target
	def encoder(self):
		cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=10) 
		outputs, h1 = tf.nn.dynamic_rnn(cell, self.embedded_gno,dtype=tf.float32)
		return outputs,h1
	def decoder(self,mode):
		cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=10)
		#output_layer = tf.layers.Dense(5,kernel_initializer=tf.truncated_normal_initializer(mean=0.1,stddev=0.1))
		output_layer = tf.layers.Dense(3)
		max_len=tf.constant(4,dtype=tf.int32)
		if self.mode=='predict':
			start_tokens = tf.tile(tf.constant([0],dtype=tf.int32),[3],name='start_token')
			helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(self.embeddings_var,start_tokens,3)
			predicting_decoder = tf.contrib.seq2seq.BasicDecoder(cell,helper,self.h1,output_layer)
			predicting_decoder_output,_,_ = tf.contrib.seq2seq.dynamic_decode(predicting_decoder,
										impute_finished = True,
										maximum_iterations = max_len)
			out=predicting_decoder_output
		if self.mode=='training':
			training_helper = tf.contrib.seq2seq.TrainingHelper(inputs = self.embedded_target,
                                                            sequence_length = self.target_len,
                                                           time_major = False)
			training_decoder = tf.contrib.seq2seq.BasicDecoder(cell,training_helper,encoder_state,output_layer)
			training_decoder_output,_,_ = tf.contrib.seq2seq.dynamic_decode(training_decoder,impute_finished=True,
                                                                        maximum_iterations = self.target_len)
			out=training_decoder_output
		return  out
	def train(self):
		x=[[2,1,2,3],[2,2,1,1],[2,2,1,1]]
		x_len=[4,4,4]
		y=[[0,1,2,3],[0,2,1,1],[0,2,1,1]]
		y_len=[4,4,4] 
		with tf.Session() as sess:
			tf.get_variable_scope().reuse_variables()
			sess.run(tf.global_variables_initializer())
			result=sess.run([self.trained_gno,self.predict_gno,self.result],feed_dict={self.inputs:x,self.target:y,
					self.inputs_len:x_len,self.target_len:y_len})
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

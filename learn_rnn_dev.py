# -*- coding: UTF-8 -*-
import tensorflow as tf
import numpy as np
from tensorflow.contrib.seq2seq import *
from tensorflow.python.layers.core import Dense
class model():
	def __init__(self,embedding_dim,rnn_size):
		#self.mode=mode
		max_target_len=4
		self.rnn_size=rnn_size
		self.index_size = 10
		self.max_len=tf.constant(max_target_len,dtype=tf.int32)
		self.embedding_dim=embedding_dim
		self.inputs = tf.placeholder(tf.int32, [None,None])
		self.batch_size= tf.placeholder(tf.int32,[None])
		self.encoder_input = tf.placeholder(tf.int32, [None,None])
		self.target = tf.placeholder(tf.int32, [None,None])
		self.inputs_len = tf.placeholder(tf.int32, [None,])
		self.target_len = tf.placeholder(tf.int32, [None,])
		self.embedded_gno = tf.contrib.layers.embed_sequence(self.inputs,self.index_size,self.embedding_dim)
		self.embeddings_var = tf.get_variable("embedding_var", [self.index_size,self.embedding_dim])
		#self.embeddings_var = tf.Variable(tf.truncated_normal(shape=[5, 2], stddev=0.1),name='encoder_embedding')
		#self.embedded_gno = tf.nn.embedding_lookup(self.embeddings_var,self.inputs )
		self.embedded_target = tf.nn.embedding_lookup(self.embeddings_var,self.encoder_input)
		self.encoded_gno,self.h1 = self.encoder()
                self.trained_gno,self.predict_gno=self.decoder()	
		self.loss = self.eva() 
	def eva(self):
		training_logits = tf.identity(self.trained_gno.rnn_output,'logits')
		masks = tf.sequence_mask(self.target_len,self.max_len,dtype=tf.float32,name="masks")
		cost = tf.contrib.seq2seq.sequence_loss(training_logits,self.target,masks)	
		#return masks,self.target
		optimizer = tf.train.AdamOptimizer()
		#gradients = optimizer.compute_gradients(cost)
		#capped_gradients = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in gradients if grad is not None]
		#train_op = optimizer.apply_gradients(capped_gradients)
		train_op=optimizer.minimize(cost)
		return  train_op,cost
	def encoder(self):
		cell = tf.contrib.rnn.LSTMCell(self.rnn_size,initializer=tf.random_uniform_initializer(-0.1,0.1,seed=2))
		#cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=self.rnn_size) 
		outputs, h1 = tf.nn.dynamic_rnn(cell, self.embedded_gno,sequence_length=self.inputs_len,dtype=tf.float32)
		return outputs,h1
	def decoder(self):
		cell = tf.contrib.rnn.LSTMCell(self.rnn_size,initializer=tf.random_uniform_initializer(-0.1,0.1,seed=2))
		#cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=self.rnn_size)
		#output_layer = tf.layers.Dense(5,kernel_initializer=tf.truncated_normal_initializer(mean=0.1,stddev=0.1))
		output_layer = tf.layers.Dense(self.index_size)
		start_tokens = tf.tile(tf.constant([0],dtype=tf.int32),self.batch_size,name='start_token')
		helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(self.embeddings_var,start_tokens,tf.constant(1))
		#predicting_decoder = tf.contrib.seq2seq.BeamSearchDecoder(cell,helper,self.h1,output_layer)
		predicting_decoder = tf.contrib.seq2seq.BasicDecoder(cell,helper,self.h1,output_layer)
		predicting_decoder_output,_,_ = tf.contrib.seq2seq.dynamic_decode(predicting_decoder,
										impute_finished = True,
										maximum_iterations = self.max_len)
		out_predict=predicting_decoder_output
		
	
		training_helper = tf.contrib.seq2seq.TrainingHelper(inputs = self.embedded_target,
                                                           sequence_length = self.target_len,
                                                          time_major = False)
		training_decoder = tf.contrib.seq2seq.BasicDecoder(cell,training_helper,self.h1,output_layer)
		training_decoder_output,_,_ = tf.contrib.seq2seq.dynamic_decode(training_decoder,impute_finished=True,
                                                                        maximum_iterations = self.max_len)
		out_train=training_decoder_output
		return  out_train,out_predict
	def train(self):
		#index[0] == 'GO'   ,index[1] == 'EOS'
		x=[[2,3,3],[2,4,2],[3,4,2]]   
		x_len=[3,3,3]
		y=[[0,2,5,8],[0,2,6,8],[0,3,7,9]]
		y_len=[4,4,4]
		y_target=[[2,5,8,1],[2,6,8,1],[3,7,9,1]]
		with tf.Session() as sess:
			tf.get_variable_scope().reuse_variables()
			sess.run(tf.global_variables_initializer())
			for i in range(1,10):
				result=sess.run([self.trained_gno,self.loss],
					feed_dict={
					self.batch_size:[len(x)],
					self.inputs:x,self.encoder_input:y,
					self.target:y_target,
					self.inputs_len:x_len,self.target_len:y_len})
				print result
			checkpoint = "data/trained_model.ckpt"
                	saver = tf.train.Saver()
                	saver.save(sess, checkpoint)
                	print('Model Trained and Saved')

	def predict(self):
                x=[[2,3,3],[4,3,2],[2,4,2]]
                x_len=[3,3,3]
                with tf.Session() as sess:
			checkpoint = "data/trained_model.ckpt"
			saver=tf.train.Saver()
			saver.restore(sess,checkpoint)
                        tf.get_variable_scope().reuse_variables()
                        result=sess.run([self.predict_gno],feed_dict={self.batch_size:[len(x)],self.inputs:x,
                                        self.inputs_len:x_len})
                        print result
                        #checkpoint = "data/trained_model.ckpt"
                        #saver = tf.train.Saver()
                        #saver.save(sess, checkpoint)
                        #print('Model Trained and Saved')


if __name__== '__main__':
	encoder=model(embedding_dim=128,rnn_size=128)
	encoder.train()
	encoder.predict()
'''
inputs = tf.placeholder(tf.float32, [None,4,1])
cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=10)
outputs, h1 = tf.nn.dynamic_rnn(cell, inputs,dtype=tf.float32)
x=[[[0],[1],[2],[3]]]
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	print sess.run([outputs,h1],feed_dict={inputs:x})
	
'''

# -*- coding: UTF-8 -*-
import tensorflow as tf
import numpy as np
from tensorflow.contrib.seq2seq import *
from tensorflow.python.layers.core import Dense
class model():
	def __init__(self,embedding_dim,rnn_size):
		#self.mode=mode
		max_target_len=5
		self.rnn_size=rnn_size
		self.index_size = 10
		self.index_sizes = tf.to_int32(tf.placeholder(tf.int32,[None]),name="index_size")
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
                self.trained_gno=self.decoder("training")
		self.predict_gno=self.decoder("predict")		
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
		return  train_op,cost,masks
	def encoder(self):
		cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=self.rnn_size) 
		outputs, h1 = tf.nn.dynamic_rnn(cell, self.embedded_gno,sequence_length=self.inputs_len,dtype=tf.float32)
		return outputs,h1
	def decoder(self,mode):
		cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=self.rnn_size)
		#output_layer = tf.layers.Dense(5,kernel_initializer=tf.truncated_normal_initializer(mean=0.1,stddev=0.1))
		output_layer = tf.layers.Dense(self.index_size)
		if mode=='predict':
			start_tokens = tf.tile(tf.constant([0],dtype=tf.int32),self.batch_size,name='start_token')
			helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(self.embeddings_var,start_tokens,1)
			predicting_decoder = tf.contrib.seq2seq.BasicDecoder(cell,helper,self.h1,output_layer)
			predicting_decoder_output,_,_ = tf.contrib.seq2seq.dynamic_decode(predicting_decoder,
										impute_finished = True,
										maximum_iterations = self.max_len)
			out=predicting_decoder_output
			#out=start_tokens
		if mode=='training':
			training_helper = tf.contrib.seq2seq.TrainingHelper(inputs = self.embedded_target,
                                                            sequence_length = self.target_len,
                                                           time_major = False)
			training_decoder = tf.contrib.seq2seq.BasicDecoder(cell,training_helper,self.h1,output_layer)
			training_decoder_output,_,_ = tf.contrib.seq2seq.dynamic_decode(training_decoder,impute_finished=True,
                                                                        maximum_iterations = self.max_len)
			out=training_decoder_output
		return  out
	def train(self):
		#index[0] == 'GO'   ,index[1] == 'EOS'
		x=[[2,3,4,1,1],[2,3,2,1,1],[2,2,2,1,1]]   
		x_len=[5,5,5]
		y=[[0,2,5,7,1],[0,2,5,5,1],[0,2,4,4,1]]
		y_len=[5,5,5]
		y_target=[[2,5,7,1,1],[2,5,5,1,1],[2,4,4,1,1]]
		with tf.Session() as sess:
			tf.get_variable_scope().reuse_variables()
			sess.run(tf.global_variables_initializer())
			for i in range(1,100):
				result=sess.run([self.trained_gno,self.loss],
					feed_dict={self.index_sizes:[10],
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
                x=[[2,3,4,1,1],[2,3,2,1,1],[2,2,2,1,1]]
                x_len=[5,5,5]
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
	encoder=model(embedding_dim=256,rnn_size=256)
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

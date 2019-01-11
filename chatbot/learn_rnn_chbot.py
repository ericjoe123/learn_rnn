# -*- coding: UTF-8 -*-
import tensorflow as tf
import numpy as np
from tensorflow.contrib.seq2seq import *
from tensorflow.python.layers.core import Dense
from datetime import datetime,timedelta
import pandas as pd
import sys
import mysql.connector
from random import randint
import time
import t
class model():
	def __init__(self,embedding_dim,rnn_size):
		#self.mode=mode
		self.rnn_size=rnn_size
		with tf.variable_scope("Input_Layer"):
			self.x_data,self.y_data,total_data=t.load_data() #read file
			self.int_to_vocab,self.vocab_to_int=self.get_dict(total_data)#build the dict of vocav2int and  int2vocab 
			self.index_size=len(self.vocab_to_int)	
			self.max_len=tf.placeholder(tf.int32,shape=(),name="max_len")
			self.inputs = tf.placeholder(tf.int32, [None,None])
			self.batch_size= tf.placeholder(tf.int32,[None])
			self.encoder_input = tf.placeholder(tf.int32, [None,None])
			self.target = tf.placeholder(tf.int32, [None,None])
			self.inputs_len = tf.placeholder(tf.int32, [None,])
			self.target_len = tf.placeholder(tf.int32, [None,])
		with tf.variable_scope("Embedding_Layer"):
			self.embedding_dim=embedding_dim
			self.embedded_gno = tf.contrib.layers.embed_sequence(self.inputs,self.index_size,self.embedding_dim)
			self.embeddings_var = tf.get_variable("embedding_var", [self.index_size,self.embedding_dim])
			#self.embeddings_var = tf.Variable(tf.truncated_normal(shape=[5, 2], stddev=0.1),name='encoder_embedding')
			#self.embedded_gno = tf.nn.embedding_lookup(self.embeddings_var,self.inputs )
			self.embedded_target = tf.nn.embedding_lookup(self.embeddings_var,self.encoder_input)
		self.encoded_gno,self.h1 = self.encoder()
                self.trained_gno,self.predict_gno=self.decoder()	
		self.loss = self.eva()
		 
	def get_dict(self,data):
	        special_words = ['<PAD>','<GO>','<EOS>']
	        #set_words = list(set([character for line in data.split('\n') for character in line]))
	        set_words=data
	        int_to_vocab = {idx:word for idx,word in enumerate(set_words+special_words)}
	        vocab_to_int = {word:idx for idx,word in int_to_vocab.items()}

        	return int_to_vocab,vocab_to_int
	def fill_pad(self,batch,batch_type):
		seq_len=[]
		result=[]
		for seq in batch:
			seq_len.append(len(seq))
		max_len=max(seq_len)
		seq_len=[]
		for seq in batch:
			if batch_type=="x":
				result.append(seq+(max_len-len(seq))*['<PAD>'])
				seq_len.append(max_len)
			if batch_type=="y":
				result.append(['<GO>']+seq+(max_len-len(seq))*['<PAD>'])
				seq_len.append(len(seq)+1)
			if batch_type=="target":
				result.append(seq+['<EOS>']+(max_len-len(seq))*['<PAD>'])
				seq_len.append(len(seq)+1)	
		return result,seq_len
	def transform_to_int(self,batch):
		result=[]
		for seq in batch:
			for word_index in range(len(seq)):
				seq[word_index]=self.vocab_to_int[seq[word_index]]
			result.append(seq)
		return result
	def process_target(self,target):
		for seq_index in range(len(target)):
			
			target[seq_index]=target[seq_index]+[self.vocab_to_int['<EOS>']]*1
			pass
		return target
	def process_y(self,y):
                for seq_index in y:
                        #y[seq_index]=[self.vocab_to_int['<GO>']]*1+y[seq_index]
			pass
		return y
	def encoder(self):
		with tf.variable_scope("Encoder"):
			cell = tf.contrib.rnn.LSTMCell(self.rnn_size,initializer=tf.random_uniform_initializer(-0.1,0.1,seed=2))
			#cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=self.rnn_size) 
			outputs, h1 = tf.nn.dynamic_rnn(cell, self.embedded_gno,sequence_length=self.inputs_len,dtype=tf.float32)
			return outputs,h1
	def decoder(self):
		with tf.variable_scope("Decoder"):
			cell = tf.contrib.rnn.LSTMCell(self.rnn_size,initializer=tf.random_uniform_initializer(-0.1,0.1,seed=2))
			#cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=self.rnn_size)
			#output_layer = tf.layers.Dense(5,kernel_initializer=tf.truncated_normal_initializer(mean=0.1,stddev=0.1))
			output_layer = tf.layers.Dense(self.index_size)
                #with tf.variable_scope("Training"):
                        training_helper = tf.contrib.seq2seq.TrainingHelper(inputs = self.embedded_target,
                                                           sequence_length = self.target_len,
                                                          time_major = False)
                        training_decoder = tf.contrib.seq2seq.BasicDecoder(cell,training_helper,self.h1,output_layer)
                        training_decoder_output,_,_ = tf.contrib.seq2seq.dynamic_decode(training_decoder,impute_finished=True,
                                                                        maximum_iterations = self.max_len)
                        out_train=training_decoder_output
		
		#with tf.variable_scope("Predicting_encoder"):
			start_tokens = tf.tile(tf.constant([self.vocab_to_int['<GO>']],dtype=tf.int32),self.batch_size,name='start_token')
			helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(self.embeddings_var,start_tokens,tf.constant(self.vocab_to_int['<EOS>']))
			#predicting_decoder = tf.contrib.seq2seq.BeamSearchDecoder(cell,helper,self.h1,output_layer)
			predicting_decoder = tf.contrib.seq2seq.BasicDecoder(cell,helper,self.h1,output_layer)
			predicting_decoder_output,_,_ = tf.contrib.seq2seq.dynamic_decode(predicting_decoder,
										impute_finished = True,
										maximum_iterations = self.max_len)
			out_predict=predicting_decoder_output
		return  out_train,out_predict
        def eva(self):
                training_logits = tf.identity(self.trained_gno.rnn_output,'logits')
                predicting_logits = tf.identity(self.predict_gno.sample_id)
		masks = tf.sequence_mask(self.target_len,self.max_len,dtype=tf.float32,name="masks")
                cost = tf.contrib.seq2seq.sequence_loss(training_logits,self.target,masks)
                #return masks,self.target
                optimizer = tf.train.AdamOptimizer()
                #gradients = optimizer.compute_gradients(cost)
                #capped_gradients = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in gradients if grad is not None]
                #train_op = optimizer.apply_gradients(capped_gradients)
                train_op=optimizer.minimize(cost)
                return  train_op,cost
	
	def train(self):
#		index[0] == 'GO'   ,index[1] == 'EOS'
		batch_size=128
		raw_input_x,input_x_len=self.fill_pad(self.x_data,batch_type="x")
		raw_input_y,input_y_len=self.fill_pad(self.y_data,batch_type="y")
		raw_input_target,input_y_len=self.fill_pad(self.y_data,batch_type="target")
	
		input_x=self.transform_to_int(raw_input_x)
		input_y=self.transform_to_int(raw_input_y)
		input_target=self.transform_to_int(raw_input_target)

		#print int_to_index
		#for index in range(0,256/batch_size):
		#	print index
		
		with tf.Session() as sess:
			tf.get_variable_scope().reuse_variables()
			sess.run(tf.global_variables_initializer())
			#writer = tf.summary.FileWriter("TensorBoard/", graph = sess.graph)
			#for index in range(0,50/batch_size):
			for index in range(0,10000/batch_size):
				i=index*batch_size
				x = list(input_x[i:i+batch_size])
                                x_len = list(input_x_len[index:index+batch_size])
                                y = list(input_y[index:index+batch_size])
                                y_len = list(input_y_len[index:index+batch_size])
                     
                                y_target = list(input_target[index:index+batch_size])
				print max(y_len)
				result = sess.run([self.trained_gno,self.loss],
					feed_dict={
					self.batch_size:[len(x)],
					self.inputs:x,self.encoder_input:y,
					self.target:y_target,
					self.inputs_len:x_len,
					self.target_len:y_len,
					self.max_len:max(y_len)})
				print result[1]
			
			checkpoint = "data/trained_model.ckpt"
                	saver = tf.train.Saver()
                	saver.save(sess, checkpoint)
                	print('Model Trained and Saved')
			print self.vocab_to_int['<GO>']
			print self.vocab_to_int['<PAD>']
			print self.vocab_to_int['<EOS>']
	'''	 			
	def predict(self):
                x=[[2,4,3,1],[2,3,2,1],[3,2,2,1]]
                x_len=[3,3,3]
                with tf.Session() as sess:
			#writer = tf.summary.FileWriter("TensorBoard/", graph = sess.graph)
			checkpoint = "data/trained_model.ckpt"
			saver=tf.train.Saver()
			saver.restore(sess,checkpoint)
                        tf.get_variable_scope().reuse_variables()
                        result=sess.run([self.predict_gno],feed_dict={self.batch_size:[len(x)],self.inputs:x,
                                        self.inputs_len:x_len,self.max_len:4})
                        print result
                        #checkpoint = "data/trained_model.ckpt"
                        #saver = tf.train.Saver()
                        #saver.save(sess, checkpoint)
                        #print('Model Trained and Saved')

	'''
if __name__== '__main__':
	#voc_size=587498
	voc_size=10
	encoder=model(embedding_dim=18,rnn_size=128)
	encoder.train()
#	encoder.predict()
'''
inputs = tf.placeholder(tf.float32, [None,4,1])
cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=10)
outputs, h1 = tf.nn.dynamic_rnn(cell, inputs,dtype=tf.float32)
x=[[[0],[1],[2],[3]]]
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	print sess.run([outputs,h1],feed_dict={inputs:x})
	
'''

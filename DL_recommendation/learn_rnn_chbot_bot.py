# -*- coding: UTF-8 -*-
import tensorflow as tf
from tensorflow.python.util import nest
import numpy as np
from tensorflow.contrib.seq2seq import *
from tensorflow.python.layers.core import Dense
from datetime import datetime,timedelta
import pandas as pd
import sys
import mysql.connector
import random
import time
import t
import jieba
from sklearn.model_selection import train_test_split
reload(sys)
sys.setdefaultencoding('utf8')

class model():
	def __init__(self,embedding_dim,rnn_size,mode):
		#with tf.name_scope("input_layer"):
		self.mode=mode
		self.beam_size=2
		self.number_layers=2
		self.rnn_size=rnn_size	
		self.x_data,self.y_data,total_data=t.load_data() #read file
		tmp_x_len=[]
		#---find the max len of x_seq---
		for x_batch in self.x_data:
			tmp_x_len.append(len(x_batch))
		self.x_max_len = max(tmp_x_len)
		print self.x_max_len
		#----------------------------
		self.int_to_vocab,self.vocab_to_int=self.get_dict(total_data)#build the dict of vocav2int and  int2vocab 
		self.index_size=len(self.vocab_to_int)	
		self.max_len=tf.placeholder(tf.int32,shape=(),name="max_len")
		self.inputs = tf.placeholder(tf.int32, [None,None])
		self.batch_size= tf.placeholder(tf.int32,[])
		self.encoder_input = tf.placeholder(tf.int32, [None,None])
		self.target = tf.placeholder(tf.int32, [None,None])
		self.inputs_len = tf.placeholder(tf.int32, [None,])
		self.target_len = tf.placeholder(tf.int32, [None,])
		#with tf.name_scope("embedding_layer"):
		self.embedding_dim=embedding_dim
		#self.embedded_gno = tf.contrib.layers.embed_sequence(self.inputs,self.index_size,self.embedding_dim)
		self.embeddings_var = tf.get_variable("embedding_var", [self.index_size,self.embedding_dim])
		#self.embeddings_var = tf.Variable(tf.truncated_normal(shape=[5, 2], stddev=0.1),name='encoder_embedding')
		self.embedded_gno = tf.nn.embedding_lookup(self.embeddings_var,self.inputs )
		self.embedded_target = tf.nn.embedding_lookup(self.embeddings_var,self.encoder_input)
		self.encoded_gno,self.h1 = self.encoder()
                self.trained_gno,self.predict_gno = self.decoder()	
		self.loss , self.loss_training = self.optimize("op")
		self.eva_loss, self.loss_testing =self.optimize("eva") 
	def get_dict(self,data):
	        special_words = ['<PAD>','<GO>','<EOS>','<None>']
	        #set_words = list(set([character for line in data.split('\n') for character in line]))
	        set_words=data
	        int_to_vocab = {idx:word for idx,word in enumerate(special_words+set_words)}
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
				x = seq+(self.x_max_len-len(seq))*['<PAD>']
				result.append(x)
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
			tmp=[]
			for word_index in range(len(seq)):
				try:
					tmp.append(self.vocab_to_int[seq[word_index]])
					#seq[word_index]=self.vocab_to_int[seq[word_index]]
				except:
					tmp.append(self.vocab_to_int['<None>'])
			result.append(tmp)
		return result
        def transform_to_vocab(self,batch):
                result=[]
		
                for seq in batch:
			tmp=[]
                        for word_index in range(len(seq)):
				#print seq[word_index]	
				if self.int_to_vocab[seq[word_index]]!="<EOS>" and seq[word_index] !=0 and seq[word_index] !=3 :
					tmp.append(self.int_to_vocab[seq[word_index]])
				if self.int_to_vocab[seq[word_index]]=="<EOS>":
					break
				#seq[word_index]=self.int_to_vocab[seq[word_index]]
                        result.append(tmp)
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
	    
        def create_rnn_cell(self):
		def single_rnn_cell():
            		single_cell = tf.contrib.rnn.LSTMCell(self.rnn_size)
            		#single_cell = tf.contrib.rnn.DropoutWrapper(single_cell, output_keep_prob=0.8)
            		return single_cell
        	cell = tf.contrib.rnn.MultiRNNCell([single_rnn_cell() for _ in range(self.number_layers)])
        	return cell
	def encoder(self):
		with tf.variable_scope("Encoder"):
			cell=self.create_rnn_cell()
			#cell  = tf.contrib.rnn.LSTMCell(self.rnn_size) 
			outputs, h1 = tf.nn.dynamic_rnn(cell, self.embedded_gno,sequence_length=self.inputs_len,dtype=tf.float32)
			return outputs,h1
	def decoder(self):
                #with tf.variable_scope("Decoder"):
               	if True:
			with tf.variable_scope('encoder'):         
				output_layer = tf.layers.Dense(self.index_size)
				decoder_cell = self.create_rnn_cell()
				'''				
				training_helper = tf.contrib.seq2seq.TrainingHelper(inputs = self.embedded_target,
                        	                                   sequence_length = self.target_len,
                        	                                  time_major = False)
                        	training_decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell,training_helper, self.h1,output_layer)
                        	training_decoder_output,_,_ = tf.contrib.seq2seq.dynamic_decode(training_decoder,impute_finished=True,
                        	                                                maximum_iterations = self.max_len)
				'''
				'''
   				encoder_outputs = tf.contrib.seq2seq.tile_batch(self.encoded_gno, multiplier=self.beam_size)
                        	encoder_state = nest.map_structure(lambda s: tf.contrib.seq2seq.tile_batch(s, self.beam_size), self.h1)
                        	#encoder_state=tf.contrib.seq2seq.tile_batch(self.h1, multiplier=self.beam_size)
                        	encoder_inputs_length = tf.contrib.seq2seq.tile_batch(self.inputs_len, multiplier=self.beam_size)
				'''
				
				attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(num_units=self.rnn_size ,memory=self.encoded_gno ,
        	                                                             memory_sequence_length=self.inputs_len) 
				attention_cell = tf.contrib.seq2seq.AttentionWrapper(decoder_cell, attention_mechanism=attention_mechanism,
               	 	                                              attention_layer_size=self.rnn_size, name='Attention_Wrapper')
				decoder_initial_state = attention_cell.zero_state(self.batch_size, dtype=tf.float32).clone(cell_state=self.h1)
				
        	                training_helper = tf.contrib.seq2seq.TrainingHelper(inputs = self.embedded_target,
        	                                                   sequence_length = self.target_len,
        	                                                  time_major = False)
        	                training_decoder = tf.contrib.seq2seq.BasicDecoder(attention_cell,training_helper, decoder_initial_state,output_layer)
        	                training_decoder_output,_,_ = tf.contrib.seq2seq.dynamic_decode(training_decoder,impute_finished=True,
        	                                                                maximum_iterations = self.max_len)  				
				out_train=training_decoder_output
				
				'''
				start_tokens = tf.tile(tf.constant([self.vocab_to_int['<GO>']],dtype=tf.int32),[self.batch_size],name='start_token')
				end_tokens = self.vocab_to_int['<EOS>']
				
				greedy_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(self.embeddings_var,start_tokens,end_tokens)
				predicting_decoder = tf.contrib.seq2seq.BasicDecoder(attention_cell,greedy_helper,decoder_initial_state,output_layer)
				predicting_decoder_output,_,_ = tf.contrib.seq2seq.dynamic_decode(predicting_decoder,
												impute_finished = True,
											maximum_iterations = self.max_len)
				out_predict=predicting_decoder_output
				'''
			
			with tf.variable_scope('encoder', reuse=True):
                                encoder_outputs = tf.contrib.seq2seq.tile_batch(self.encoded_gno, multiplier=self.beam_size)
                                encoder_state = nest.map_structure(lambda s: tf.contrib.seq2seq.tile_batch(s, self.beam_size), self.h1)
                                #encoder_state=tf.contrib.seq2seq.tile_batch(self.h1, multiplier=self.beam_size)
                                encoder_inputs_length = tf.contrib.seq2seq.tile_batch(self.inputs_len, multiplier=self.beam_size)
	
				attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(num_units=self.rnn_size ,memory=encoder_outputs,
                                                                   memory_sequence_length=encoder_inputs_length)
	                	attention_cell = tf.contrib.seq2seq.AttentionWrapper(decoder_cell, attention_mechanism=attention_mechanism,
                                                                      attention_layer_size=self.rnn_size, name='Attention_Wrapper')
                        	decoder_initial_state = attention_cell.zero_state(self.batch_size*self.beam_size, dtype=tf.float32).clone(cell_state=encoder_state)
			
				start_tokens = tf.tile(tf.constant([self.vocab_to_int['<GO>']],dtype=tf.int32),[self.batch_size],name='start_token')
	                	end_tokens = self.vocab_to_int['<EOS>']

				beam_decoder = tf.contrib.seq2seq.BeamSearchDecoder(attention_cell, embedding=self.embeddings_var,
                                                                             start_tokens=start_tokens, end_token=end_tokens,
                                                                             initial_state=decoder_initial_state,
                                                                             beam_width=self.beam_size,
                                                                             output_layer=output_layer)
				beam_decoder_output,_,_ = tf.contrib.seq2seq.dynamic_decode(beam_decoder,
                                                                               maximum_iterations = self.max_len)
			
				out_predict = beam_decoder_output
			
				
		return  out_train,out_predict
        def optimize(self,op):
                #predict_logits = tf.identity(self.trained_gno.rnn_output,'predict')
		training_logits = tf.identity(self.trained_gno.rnn_output,'logits')
                #predicting_logits = tf.identity(self.predict_gno)
		masks = tf.sequence_mask(self.target_len,self.max_len,dtype=tf.float32,name="masks")
                
		#cost = tf.contrib.seq2seq.sequence_loss(training_logits,self.target,masks)
		#tf.summary.scalar('loss', cost)
		if op=="op":
			cost_training = tf.contrib.seq2seq.sequence_loss(training_logits,self.target,masks)
			#return masks,self.target
			global lr
                	optimizer = tf.train.AdamOptimizer(0.01)
                	#gradients = optimizer.compute_gradients(cost_training)
                	#capped_gradients = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in gradients if grad is not None]
                	#train_op = optimizer.apply_gradients(capped_gradients)
                	train_op=optimizer.minimize(cost_training)
                	loss_training=tf.summary.scalar('loss_training', cost_training)
			return  [train_op,cost_training],loss_training
		elif op=="eva":
			cost_testing = tf.contrib.seq2seq.sequence_loss(training_logits,self.target,masks)
			loss_testing=tf.summary.scalar('loss_testing', cost_testing)
			return cost_testing,loss_testing
	
	def train(self):
		checkpoint = "data_qa/trained_model.ckpt"
		batch_size=128
		step=0
		with tf.Session() as sess:
			try:
                        	saver=tf.train.Saver()
                        	saver.restore(sess,checkpoint)
				tf.get_variable_scope().reuse_variables()
			except:
				
				tf.get_variable_scope().reuse_variables()
				sess.run(tf.global_variables_initializer())
			#merged = tf.summary.merge_all()
			writer = tf.summary.FileWriter("TensorBoard/", graph = sess.graph)
			total_data_len=len(self.x_data)
			global lr
			lr=0.7
			epoch=1000
			loss=1000
			
			for index in range(0,len(self.x_data)):
				self.x_data[index]=self.x_data[index][::-1]
			
			#random_select=False
			#if random_select==True:
			#	cut_index=random.randint(0,(total_data_len/batch_size)-1)
			#else:
			cut_index=(total_data_len/batch_size)-1
			start=cut_index*batch_size
			stop=cut_index*batch_size+batch_size
			x_data_training=self.x_data[:start]+self.x_data[stop:]
			y_data_training=self.y_data[:start]+self.y_data[stop:]
			#x_data_training=self.x_data
			#y_data_training=self.y_data
			print len(self.x_data)
			print len(x_data_training)
			batch_x_test=self.x_data[start:stop]
			batch_y_test=self.y_data[start:stop]
			for epoch_times in range(0,epoch):
				if epoch_times>=20:
					lr=0.35
				if epoch_times>50:
					lr=0.175
			#while loss>=0.2:	
				tmp=zip(x_data_training,y_data_training)
                		random.shuffle(tmp)
                		source_data_x, source_data_y = zip(*tmp)
				for index in range(0,total_data_len/batch_size):
					i=index*batch_size
					batch_x = list(source_data_x[i:i+batch_size])
                                	batch_y = list(source_data_y[i:i+batch_size])
					raw_input_x,x_len=self.fill_pad(batch_x,batch_type="x")
		                	raw_input_y,y_len=self.fill_pad(batch_y,batch_type="y")
                			raw_input_target,y_target_len=self.fill_pad(batch_y,batch_type="target")
					#print y_len
                			x=self.transform_to_int(raw_input_x)
                			y=self.transform_to_int(raw_input_y)
		                	y_target=self.transform_to_int(raw_input_target)
					#print max(x_len)	
					if self.mode=="training":
						training_result = sess.run([self.loss],
						#result = sess.run([self.loss],
						feed_dict={
						self.batch_size:len(x),
						self.inputs:x,self.encoder_input:y,
						self.target:y_target,
						self.inputs_len:x_len,
						self.target_len:y_len,
						self.max_len:max(y_len)})
						
						x_test,x_len_test=self.fill_pad(batch_x_test,batch_type="x")
						y_test,y_len_test=self.fill_pad(batch_y_test,batch_type="y")
						y_target_test,y_target_len_test=self.fill_pad(batch_y_test,batch_type="target")
                                        	x_test=self.transform_to_int(x_test)
                                        	y_test=self.transform_to_int(y_test)
                                        	y_target_test=self.transform_to_int(y_target_test)
	
						test_result = sess.run([self.eva_loss],
                                                feed_dict={
                                                self.batch_size:len(x_test),
                                                self.inputs:x_test,self.encoder_input:y_test,
                                                self.target:y_target_test,
                                                self.inputs_len:x_len_test,
                                                self.target_len:y_len_test,
                                                self.max_len:max(y_len_test)})
					
						print "epoch : "+str(epoch_times)
						print "batch : "+str(i)+"~"+str(i+batch_size)
						print "training loss: "+str(training_result[0])
						print "testing loss: "+str(test_result[0])
						print "total word number: "+str(len(self.int_to_vocab))
						print "step: "+str(step)
					
						print lr
					        if step%20==0:
                                                	saver = tf.train.Saver()
                                                	saver.save(sess, checkpoint)
                                                	result_graph = sess.run(self.loss_training,
                                                	feed_dict={
                                                	self.batch_size:len(x),
                                                	self.inputs:x,self.encoder_input:y,
                                                	self.target:y_target,
                                                	self.inputs_len:x_len,
                                                	self.target_len:y_len,
                                                	self.max_len:max(y_len)})
                                                	#result_graph=result[0]
                                                	writer.add_summary(result_graph, step)
                                                	result_graph2 = sess.run(self.loss_testing,
                                                	feed_dict={
                                                	self.batch_size:len(x_test),
                                                	self.inputs:x_test,self.encoder_input:y_test,
                                                	self.target:y_target_test,
                                                	self.inputs_len:x_len_test,
                                                	self.target_len:y_len_test,
                                                	self.max_len:max(y_len_test)})
                                                	#result_graph=result[0]
                                                	writer.add_summary(result_graph2, step)
                                                	print "saved"

                          

					
					if self.mode=='beam':
						result = sess.run([self.predict_gno],
                                        	#result = sess.run([self.loss],
                                                feed_dict={
                                                self.batch_size:len(x),
                                                self.inputs:x,self.encoder_input:y,
                                                self.target:y_target,
                                                self.inputs_len:x_len,
                                                self.target_len:y_len,
                                                self.max_len:max(y_len)})
								
						#output=result[0].sample_id
						#out=self.transform_to_vocab(output)
						#predict=result[0].sample_id
                                        	#pred=self.transform_to_vocab(predict)
						#print result[1]
						
						predict_beam=result[0].predicted_ids
						print len(predict_beam)
						#=======fixing the beam_output to [ [x1,x2,x,...xn],[x1,x2...]...] ========
						beam_output_1=[]
						beam_output_2=[]
						for seq_ids in predict_beam:
							tmp_1=[]
							tmp_2=[]
							for ids in seq_ids:		
								if ids[0] >0:
									tmp_1.append( ids[0])
								if ids[1] > 0:
									tmp_2.append( ids[1])
							beam_output_1.append(tmp_1)
							beam_output_2.append(tmp_2)
						#=========================================================================	
						#print beam_output
						pred1=self.transform_to_vocab(beam_output_1)
			 			pred2=self.transform_to_vocab(beam_output_2)
						
						for i in range(len(raw_input_x)):
							#print x[i]
							#print raw_input_x[i]
							#print raw_input_y[i]
							#print raw_input_target[i]
							print "問句 : "+''.join(batch_x[i][::-1])
							#print "機器人回答(traniing) : "+''.join(out[i])
							#print "機器人回答(predicting) : "+''.join(pred[i])
							print "機器人回答(predicting) : "+''.join(pred1[i])
							print "機器人回答(predicting) : "+''.join(pred2[i])
							#print predict_beam[i]
							#print predict[i]
							#print output[i]
							print "鄉民回答 : "+''.join(batch_y[i])
							
							print "============================="
                				
								
					step+=1
			saver = tf.train.Saver()
                        saver.save(sess, checkpoint)
                        print('Model Trained and Saved')
		 			
	def predict(self,x):
                #x=[[2,4,3,1],[2,3,2,1],[3,2,2,1]]
                #x_len=[3,3,3]
                with tf.Session() as sess:
			#writer = tf.summary.FileWriter("TensorBoard/", graph = sess.graph)
			checkpoint = "data_qa/trained_model.ckpt"
			saver=tf.train.Saver()
			saver.restore(sess,checkpoint)
                        tf.get_variable_scope().reuse_variables()
			x=x[0:35]
			x=[x[0][::-1]]
				
			x_pad=self.fill_pad(x,"x")
			x_pad=self.transform_to_int(x_pad[0])
			print x_pad
			x=x_pad
			print x
			x_len=[len(x[0])]
			
			result=sess.run([self.predict_gno,self.embedded_gno],feed_dict={self.batch_size:[len(x)],self.inputs:x,
                                       self.inputs_len:x_len,self.max_len:len(x[0])})
                       	output=result[0][1]
			#print result
			out=self.transform_to_vocab(output)
			print ''.join(out[0])
			print 
			return ''.join(out[0])
			#count+=1
			#if count>10:
			#	bot_working=False	
                        #checkpoint = "data/trained_model.ckpt"
                        #saver = tf.train.Saver()
                        #saver.save(sess, checkpoint)
                        #print('Model Trained and Saved')

				
if __name__== '__main__':
	#voc_size=587498
	mode=sys.argv[1]
	voc_size=10
	global lr
	lr=0.7
	seq2seq=model(embedding_dim=128,rnn_size=256,mode=mode)
	seq2seq.train()
	#encoder.predict([[1969, 1793, 1867, 402, 2731, 0, 0, 0, 0, 0, 0, 0, 0]])
	#encoder.predict([[1969, 1793, 1867, 402, 2731, 0, 0, 0]])
	#encoder.predict([[856, 4088, 651, 6724, 1864, 4785, 5666, 2262, 0, 0, 0, 0, 0, 0, 0]])
	#encoder.predict([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3560, 4884, 2752, 5617, 6670, 5930, 774, 291]])
	while True:
		x=raw_input()
		#print str(x)
		x=str(x).decode('utf8')
		tmp=jieba.cut(x, cut_all=False)
		input_x=[]
		for word in tmp:
			input_x.append(word)
		#encoder.predict([x])
		encoder.predict([input_x])
'''
inputs = tf.placeholder(tf.float32, [None,4,1])
cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=10)
outputs, h1 = tf.nn.dynamic_rnn(cell, inputs,dtype=tf.float32)
x=[[[0],[1],[2],[3]]]
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	print sess.run([outputs,h1],feed_dict={inputs:x})
	
'''

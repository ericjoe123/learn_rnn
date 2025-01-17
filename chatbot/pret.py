# -*- coding: UTF-8 -*-
import tensorflow as tf
import numpy as np
from collections import Counter
import jieba
def  load_data():
	with open('data_bot.txt','r') as file:
		data=file.readlines()
	result=[]
	total=[]
	Q=[]
	A=[]
	for i in data:
		i=i.strip('\n')
		i=i.split('\t')	
		#i[0]=i[0].decode('utf8')
		#i[1]=i[1].decode('utf8')
		tmp_q=[]
		tmp_a=[]
		i[0] = jieba.cut(i[0], cut_all=False)
		i[1] = jieba.cut(i[1], cut_all=False)
		for word in i[0]:
			if word != ' ':
				tmp_q.append(word.encode('utf8'))
				total.append(word)
		tmp_q='<,>'.join(tmp_q)
		for word in i[1]:
			if word != ' ':
				tmp_a.append(word.encode('utf8'))
				total.append(word)
		tmp_a='<,>'.join(tmp_a)
		Q.append(tmp_q+'\t'+tmp_a+'\n')
	Q=''.join(Q)
		#A.append(tmp_a)
		
	#for word in total:
	#	counter=Counter(total)
	#	for word in counter:
	#		if counter[word]>1:
	#			result.append(word)
	#total=result
	return Q,list(set(total))
	#print Q
	#print "==================="
	#print A
def extract_character_vocab(data):
	
    	special_words = ['<PAD>','<GO>','<EOS>']
    	#set_words = list(set([character for line in data.split('\n') for character in line]))
	set_words=data
	int_to_vocab = {idx:word for idx,word in enumerate(set_words+special_words)}
    	vocab_to_int = {word:idx for idx,word in int_to_vocab.items()}

    	return int_to_vocab,vocab_to_int
if __name__ == '__main__':
	Q,data=load_data()
	with open('preprocessed_data','w') as f:
		f.write(Q)
	print data
	print len(data)
	print len(Q)
	#for i in range(len(Q)):
	#	print Q[i]
	#	print "========================"
	#print data
	#int_to_vocab,vocab_to_int=extract_character_vocab(data)
	#for i in int_to_vocab:
	#	print int_to_vocab[i] , i
	
	#print len(int_to_vocab)

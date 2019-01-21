#-*- coding: utf-8 -*-
import requests 
from bs4 import BeautifulSoup
#from urllib.request import urlretrieve
from skpy import SkypeEventLoop, SkypeNewMessageEvent
import sys
import re
from datetime import datetime,timedelta
from random import randint
import requests
import jieba.posseg as pseg
import math
import json
from rediscluster import StrictRedisCluster
from collections import Counter
reload(sys)
sys.setdefaultencoding('utf8')
from skpy import Skype
import pymysql
import random
import learn_rnn_chbot_bot as seq2seq
import jieba
class SkypePing(SkypeEventLoop):
	def __init__(self):
        	super(SkypePing, self).__init__('ericjoe123@hotmail.com', 'Ggyyhope3')
   		self.model=seq2seq.model(embedding_dim=18,rnn_size=128)	
    	def onEvent(self, event):
        	if isinstance(event, SkypeNewMessageEvent) and not event.msg.userId == self.userId and "小幫手_" in event.msg.content :
			term=event.msg.content
                        term=term.split('_')
                        word=term[1]
			x=str(word).decode('utf8')
			tmp=jieba.cut(x, cut_all=False)
			input_x=[]
        	        for word in tmp:
        	               input_x.append(word)
           	        seq=self.model.predict([input_x])
			#event.msg.chat.sendMsg(''.join(word))
			
			#event.msg.chat.sendMsg(seq)
			
		if isinstance(event, SkypeNewMessageEvent) and not event.msg.userId == self.userId and ("商品數量_" in event.msg.content or "gnum_" in event.msg.content):
		       	term=event.msg.content
			term=term.split('_')
			word=term[1]
			try:
				date=re.sub(r'\D', "", word)
				date=date[0:4]+'-'+date[4:6]+'-'+date[6:8]	
				event.msg.chat.sendMsg('正在嘗試讀取'+date+'之檔案')
					
				try:
					event.msg.chat.sendFile(open("/home/webuser/ericjoe/RTBreport/document/goods_B"+date+".csv", "rb"),\
					"goods_B"+date+".csv")
					event.msg.chat.sendFile(open("/home/webuser/ericjoe/RTBreport/document/goods_C"+date+".csv", "rb"),\
					"goods_C"+date+".csv")
				except:
					event.msg.chat.sendMsg('日期錯誤,請查詢2018/7/2號以後之資料')
					
						
			except:
				event.msg.chat.sendMsg("請輸入正確格格式 商品數量_日期  OR  gnum_date ")
				event.msg.chat.sendMsg("例如: gnum_20180827 商品_2018-08-27 , 商品_20180827 ,商品_2018/08/27")


		if isinstance(event, SkypeNewMessageEvent)  and ("熱賣商品" in event.msg.content or "熱門商品" in event.msg.content):
			gno_list=self.hot_item()
			index=random.randint(0,9)
			gno=str(list(gno_list)[index][1])
			name=str(list(gno_list)[index][2])
			num=str(list(gno_list)[index][0])
			link="<a href=https://goods.ruten.com.tw/item/show?"+gno+">https://goods.ruten.com.tw/item/show?"+gno+"</a>"
			event.msg.chat.sendMsg("TOP "+str(index)+" 商品 : "+name)
			event.msg.chat.sendMsg("已賣出 "+num+ "件")
			event.msg.chat.sendMsg(link)
		if isinstance(event, SkypeNewMessageEvent)and not event .msg.userId == self.userId and ("強制關閉_" in event.msg.content or  "fstop_" in event.msg.content):
			event.msg.chat.sendMsg('小幫手已離線')
			state[0]='0'
			sys.exit()
	def hot_item(self):
		db=pymysql.connect("172.25.10.167","imuser","rtnJ@LnQ2017","rtb_bot")
		cur=db.cursor () 
		cur.execute( "SELECT order_qty,g_no,g_name FROM `rtb_bot_order` WHERE r_status='ACTIVE' AND oracle_r_status='Y' ORDER BY `rtb_bot_order`.`order_qty` DESC" )
		result=cur.fetchall()
		cur.close()
		db.close()
		return result
file = open("/home/webuser/ericjoe/RTBreport/setting.txt", "rt")
f=file.readlines()
state=[]
for s in f:
	s=s.strip('\n')
	state.append(str(s))
file.close()


if state[0]=='1':
	print "start bot"
	print (datetime.today()).strftime("%Y-%m-%d")
	print datetime.today()
	state[0]='1'
	r = open("/home/webuser/ericjoe/RTBreport/setting.txt", "w")
	r.write(state[0])
	r.close()
	#print state[0]
	print "小幫手啟動中...."
	tmp=SkypePing()
	print "小幫手啟動成功"
	tmp.loop()
'''	
	try:
		tmp.loop()
	except:
		state[0]='0'
        	r = open("/home/webuser/ericjoe/RTBreport/setting.txt", "w")
        	r.write(state[0])
        	r.close()
        	print "stop"
        	print (datetime.today()).strftime("%Y-%m-%d")
        	print datetime.today()
		sys.exit()

if state[0]=='1':
        print "bot is working"
        print (datetime.today()).strftime("%Y-%m-%d")
        print datetime.today()
'''

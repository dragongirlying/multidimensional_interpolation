#!/usr/bin/python
# -*- coding: utf-8 -*-
from sklearn import linear_model
import sys
import re
import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import MinMaxScaler
from scipy.interpolate import interp1d
import copy

data=pd.read_csv(sys.argv[1])
#find the number of dimension
para_num=int((re.search(r'(\d)D.csv',sys.argv[1])).group(1))

X=np.array([data["para_1"][i] for i in range(len(data))])
Y=np.array([data["result"][i] for i in range(len(data))])

train_number=150
max=None
min=None
max_y=None
min_y=None
for i in range(train_number):
	if(max==None):
		max=X[i]
	elif(X[i]>max and max!=None):
		max=X[i]
	if(min==None):
		min=X[i]
	elif(X[i]<min and min!=None):
		min=X[i]
	if(max_y==None):
		max_y=Y[i]
	elif(Y[i]>max_y and max_y!=None):
		max_y=Y[i]
	if(min_y==None):
		min_y=Y[i]
	elif(Y[i]<min_y and min_y!=None):
		min_y=Y[i]
interval=max-min
interval_y=max_y-min_y

#Normalization
X_train=[]
Y_train=[]
for i in range(train_number):
	if ((X[i]-min)/interval) not in X_train:    #Remove the dupilicate
		X_train.append((X[i]-min)/interval)
		Y_train.append((Y[i]-min_y)/interval_y)
	else: 
		continue
#Sort
X_train_tmp=np.array(X_train)
Y_train_tmp=np.array(Y_train)
X_train=X_train_tmp[np.argsort(X_train_tmp)]
Y_train=Y_train_tmp[np.argsort(X_train_tmp)]

#三次样条
f=interp1d(X_train,Y_train,kind='cubic')
#Deal with data of prediction
pred_start=150
pred_end=200
pred_len=pred_end-pred_start
X_test = copy.deepcopy(X[pred_start:pred_end])
for i in range(len(X_test)):
	X_test[i]=(X_test[i]-min)/interval
Y_test=Y[pred_start:pred_end]
#Prediction
Y_pred = f(X_test)
for i in range(len(Y_pred)):
	Y_pred[i]=Y_pred[i]*interval_y+min_y
Y_delta=[] 
for i in range(pred_len):
	Y_delta.append(Y_pred[i]-Y_test[i])

out=open("predict_"+str(para_num)+"D_cubic.csv",'w+')
out.write("para_1,result,Y_pred,Y_delta\n")
for i in range(pred_len):
	out.write(str(X[i+pred_start])+',')
	out.write(str(Y_test[i])+','+str(Y_pred[i])+','+str(Y_delta[i])+'\n')
out.close()

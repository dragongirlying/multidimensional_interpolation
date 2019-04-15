#!/usr/bin/python
# -*- coding: utf-8 -*-
from sklearn import linear_model
import sys
import re
import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import MinMaxScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
import copy

data=pd.read_csv(sys.argv[1])
#find the number of dimension
para_num=int((re.search(r'(\d)D_new.csv',sys.argv[1])).group(1))
para_name=[]
for i in range(1,int(para_num)+1):
	para_name.append("para_"+str(i))

out=open("predict_"+str(para_num)+"D_poly.csv",'w+')
for name in para_name:
	out.write(name+",")
out.write("result,Y_pred,Y_delta\n")
X=[]
Y=[]

train_number=1000
for i in range(len(data)):
	X.append([data[name][i] for name in para_name])
	Y.append([data["result"][i]])

pred_start=train_number+1
pred_end=train_number+100
pred_len=pred_end-pred_start

for k in range(pred_len):
	X_train=[]
	Y_train=[]
	X_test = copy.deepcopy(X[pred_start+k])
	for i in range(train_number):
		flag=1
		for j in range(para_num):
			if(abs(X_test[j])*0.25<abs(X[i][j]) and abs(X_test[j])*4>abs(X[i][j])):
				flag=1
			else:
				flag=0
			if(flag==0):
				break
		if(flag==1): 
			X_train.append(X[i])
			Y_train.append(Y[i])
	if(len(X_train)==0 or len(X_train)==1):
		for i in range(train_number):
			X_train.append(X[i])
			Y_train.append(Y[i])
	max=[None]*para_num
	min=[None]*para_num
	interval=[None]*para_num
	for j in range(para_num):
		for i in range(len(X_train)):
			if(max[j]==None):
				max[j]=X_train[i][j]
			elif(X_train[i][j]>max[j] and max[j]!=None):
				max[j]=X_train[i][j]
			if(min[j]==None):
				min[j]=X_train[i][j]
			elif(X_train[i][j]<min[j] and min[j]!=None):
				
				min[j]=X_train[i][j]
		interval[j]=max[j]-min[j]
	max_y=None
	min_y=None
	for i in range(len(Y_train)):
		if(max_y==None):
			max_y=Y_train[i][0]
		elif(Y_train[i][0]>max_y and max_y!=None):
			max_y=Y_train[i][0]
		if(min_y==None):
			min_y=Y_train[i][0]
		elif(Y_train[i][0]<min_y and min_y!=None):
			min_y=Y_train[i][0]
	interval_y=max_y-min_y

	#Normalization

	min_max_scaler=MinMaxScaler()
	X_norm=min_max_scaler.fit_transform(X_train)
	Y_norm=min_max_scaler.fit_transform(Y_train)


	gpr=GaussianProcessRegressor().fit(X_norm, Y_norm)
	for j in range(para_num):
		X_test[j]=(X_test[j]-min[j])/interval[j]
	Y_test=Y[pred_start+k]
	#Prediction
	[Y_pred]= gpr.predict([X_test])
	Y_pred[0]=Y_pred[0]*interval_y+min_y
	Y_delta=Y_pred[0]-Y_test


	for j in range(para_num):
		out.write(str(X[k+pred_start][j])+',')
	out.write(str(Y_test[0])+','+str(Y_pred[0])+','+str(Y_delta[0])+'\n')
out.close()

	

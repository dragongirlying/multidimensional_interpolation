#!/usr/bin/python
# -*- coding: utf-8 -*-
from sklearn import linear_model
import sys
import re
import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import MinMaxScaler
import copy

data=pd.read_csv(sys.argv[1])
#find the number of dimension
para_num=int((re.search(r'(\d)D.csv',sys.argv[1])).group(1))
para_name=[]
for i in range(1,int(para_num)+1):
	para_name.append("para_"+str(i))

X=np.array([[data[name][i] for name in para_name] for i in range(len(data))])
Y=np.array([[data["result"][i]] for i in range(len(data))])

train_number=150
max=[None]*para_num
min=[None]*para_num
interval=[None]*para_num
for j in range(para_num):
	for i in range(train_number):
		if(max[j]==None):
			max[j]=X[i][j]
		elif(X[i][j]>max[j] and max[j]!=None):
			max[j]=X[i][j]
		if(min[j]==None):
			min[j]=X[i][j]
		elif(X[i][j]<min[j] and min[j]!=None):
			min[j]=X[i][j]
	interval[j]=max[j]-min[j]
max_y=None
min_y=None
for i in range(len(Y)):
	if(max_y==None):
		max_y=Y[i][0]
	elif(Y[i][0]>max_y and max_y!=None):
		max_y=Y[i][0]
	if(min_y==None):
		min_y=Y[i][0]
	elif(Y[i][0]<min_y and min_y!=None):
		min_y=Y[i][0]
interval_y=max_y-min_y
#Normalization
min_max_scaler=MinMaxScaler()
X_norm=min_max_scaler.fit_transform(X[:train_number])
Y_train=min_max_scaler.fit_transform(Y[:train_number])
#Poly Linear
poly=PolynomialFeatures(degree=2)
X_train=poly.fit_transform(X_norm)
#Build model
regr = linear_model.LinearRegression()
regr.fit(X_train, Y_train)
#Deal with data of prediction
pred_start=150
pred_end=200
pred_len=pred_end-pred_start
X_test = copy.deepcopy(X[pred_start:pred_end])
for i in range(len(X_test)):
	for j in range(para_num):
		X_test[i][j]=(X_test[i][j]-min[j])/interval[j]
X_test_poly=poly.fit_transform(X_test)
Y_test=Y[pred_start:pred_end]
#Prediction
Y_pred = regr.predict(X_test_poly)
for i in range(len(Y_pred)):
	Y_pred[i][0]=Y_pred[i][0]*interval_y+min_y
Y_delta=[] 
for i in range(pred_len):
	Y_delta.append(Y_pred[i]-Y_test[i])


out=open("predict_"+str(para_num)+"D_poly.csv",'w+')
for name in para_name:
	out.write(name+",")
out.write("result,Y_pred,Y_delta\n")
for i in range(pred_len):
	for j in range(para_num):
		out.write(str(X[i+pred_start][j])+',')
	out.write(str(Y_test[i][0])+','+str(Y_pred[i][0])+','+str(Y_delta[i][0])+'\n')
out.close()

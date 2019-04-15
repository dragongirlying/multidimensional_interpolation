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
from scipy.interpolate import interp2d
from scipy.interpolate import griddata
import copy

data=pd.read_csv(sys.argv[1])
#find the number of dimension
para_num=int((re.search(r'(\d)D.csv',sys.argv[1])).group(1))
para_name=[]
for i in range(1,int(para_num)+1):
	para_name.append("para_"+str(i))
X=np.array([[data[name][i] for name in para_name] for i in range(len(data))])
Y=np.array([data["result"][i] for i in range(len(data))])

train_number=150
max1=None
min1=None
max2=None
min2=None
max3=None
min3=None
max_y=None
min_y=None
for i in range(train_number):
	if(max1==None):
		max1=X[i][0]
	elif(X[i][0]>max1 and max1!=None):
		max1=X[i][0]
	if(min1==None):
		min1=X[i][0]
	elif(X[i][0]<min1 and min1!=None):
		min1=X[i][0]
	if(max2==None):
		max2=X[i][1]
	elif(X[i][1]>max2 and max2!=None):
		max2=X[i][1]
	if(min2==None):
		min2=X[i][1]
	elif(X[i][1]<min2 and min2!=None):
		min2=X[i][1]
	if(max3==None):
		max3=X[i][2]
	elif(X[i][2]>max3 and max3!=None):
		max3=X[i][2]
	if(min3==None):
		min3=X[i][2]
	elif(X[i][2]<min3 and min3!=None):
		min3=X[i][2]
	if(max_y==None):
		max_y=Y[i]
	elif(Y[i]>max_y and max_y!=None):
		max_y=Y[i]
	if(min_y==None):
		min_y=Y[i]
	elif(Y[i]<min_y and min_y!=None):
		min_y=Y[i]
interval1=max1-min1
interval2=max2-min2
interval3=max3-min3
interval_y=max_y-min_y

#Normalization
X_train=copy.deepcopy(X[:train_number])
Y_train=[]
for i in range(train_number):
	X_train[i][0]=(X[i][0]-min1)/interval1
	X_train[i][1]=(X[i][1]-min2)/interval2
	X_train[i][2]=(X[i][2]-min3)/interval3
	Y_train.append((Y[i]-min_y)/interval_y)
Y_train=np.array(Y_train)

#Deal with data of prediction
pred_start=150
pred_end=200
pred_len=pred_end-pred_start
X_test = copy.deepcopy(X[pred_start:pred_end])
for i in range(len(X_test)):
	X_test[i][0]=(X_test[i][0]-min1)/interval1
	X_test[i][1]=(X_test[i][1]-min2)/interval2
	X_test[i][2]=(X_test[i][2]-min3)/interval3
Y_test=Y[pred_start:pred_end]
#三次样条
Y_pred = griddata(X_train, Y_train, X_test, method='linear')
for i in range(len(Y_pred)):
	Y_pred[i]=Y_pred[i]*interval_y+min_y
Y_delta=[] 
for i in range(pred_len):
	Y_delta.append(Y_pred[i]-Y_test[i])

out=open("predict_"+str(para_num)+"D_cubic.csv",'w+')
for name in para_name:
	out.write(name+",")
out.write("result,Y_pred,Y_delta\n")
for i in range(pred_len):
	out.write(str(X[i+pred_start][0])+',')
	out.write(str(X[i+pred_start][1])+',')
	out.write(str(X[i+pred_start][2])+',')
	out.write(str(Y_test[i])+','+str(Y_pred[i])+','+str(Y_delta[i])+'\n')
out.close()

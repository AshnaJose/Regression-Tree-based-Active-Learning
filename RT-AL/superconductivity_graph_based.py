#Consist of code for Graph-based approach, run only till 100 samples were labeled for the superconductivity dataset as due to computational complexity.

import numpy as np
import math
from math import sqrt
from sklearn.tree import DecisionTreeRegressor 
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import preprocessing,datasets, linear_model
from sklearn.linear_model import Ridge
import copy
import scipy
import pandas as pd

from scipy.spatial.distance import cityblock

import warnings
warnings.filterwarnings('ignore')

cmax=5
maxtree=200
maxlabel1 = 15     #first n labeled by non active part
maxlabel4 = [maxlabel1]

maxlabel2=[20,40,60,80,100]

maxlabel3 = [20-maxlabel1,20,20,20,20]

error_g = np.zeros((len(maxlabel2),maxtree))

feat=81
min_samples_leaf = 3
treeseed = None

rf = RandomForestRegressor(min_samples_leaf = min_samples_leaf)

df = pd.read_csv('data_sets/supercond.csv')
X = df.iloc[:, :-1]
Y = df.iloc[:, -1]

print(len(X),len(Y))

def ClusterIndices(clustNum, labels_array): 
            return np.where(labels_array == clustNum)[0]
    
from collections import Counter, defaultdict

from sklearn.cluster import KMeans
from scipy.cluster.vq import vq

for tree in range(maxtree):
    print('Tree_',tree,flush=True)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.75,random_state=tree)
    
    X_train1 = X_train                
    Y_train1 = Y_train
    X_train1 = X_train1.to_numpy()    
    Y_train1 = Y_train1.to_numpy()
    X_train2 = X_train1                 
    
    scaler = preprocessing.StandardScaler().fit(X_train1)    #scaled
    X_train1 = scaler.transform(X_train1)

    treeseed = None

    g_MSE = np.zeros([len(maxlabel2)])
    
    X_train3 = X_train1       #scaled
    Y_train3 = Y_train1
    X_train2 = X_train2     #unscaled
    
    cv_ind = np.random.permutation(range(X_train2.shape[0]))
    
    X_train3 = X_train1[cv_ind,:]        #scaled
    Y_train3 = Y_train1[cv_ind]
    X_train2 = X_train2[cv_ind,:]        #unscaled
    X_train_new=X_train2[:maxlabel1]
    X_train_new_scaled = X_train3[:maxlabel1]
    Y_train_new=Y_train3[:maxlabel1]
    
    X_train2 = np.delete(X_train2,range(maxlabel1),axis = 0)
    X_train3 = np.delete(X_train3,range(maxlabel1),axis = 0)
    Y_train3 = np.delete(Y_train3,range(maxlabel1))
    
    K=0
    s=[]
    
    for x in X_train3:
            theta1 = np.min([cityblock(x,x_lab) for x_lab in X_train_new_scaled])
            s = np.append(s,theta1)
    z1 = np.sum(s)
                    
    for rep in range(maxlabel1,maxlabel2[len(maxlabel2)-1]):
        
        if rep>maxlabel1:
            s_new=[]
            s_new = [np.append(s_new,cityblock(x,new_labeled)) for x in X_train3]  
            s = np.append(s,s_new)
            s=s.reshape(2,-1)    
            s1 = [s[:,j].min() for j in range(len(s[0]))]
            s = s1
            z1 = np.sum(s)
        
        z=[]
        theta=0
        for x1 in X_train3:
            
            theta2=[]
            z2=0
            i=0
            for x2 in X_train3:
           
                theta2 = []
                theta2 = np.append(theta2,s[i])
                theta2 = np.append(theta2,cityblock(x2,x1))

                z2 +=np.min(theta2)
                i+=1
            z = np.append(z,z1-z2)
        theta = np.argmax(z)
        
        new_labeled = X_train3[theta]
        X_train_new = np.array(np.append(X_train_new,X_train2[theta]))
        X_train_new = np.hsplit(X_train_new,len(X_train_new)/int(feat))
        X_train_new_scaled = np.array(np.append(X_train_new_scaled,X_train3[theta]))
        X_train_new_scaled = np.hsplit(X_train_new_scaled,len(X_train_new_scaled)/int(feat))
        Y_train_new = np.append(Y_train_new,Y_train3[theta])
        X_train2 = np.delete(X_train2,theta,axis = 0)
        X_train3 = np.delete(X_train3,theta,axis = 0)
        Y_train3 = np.delete(Y_train3,theta)
        s = np.delete(s,theta)
        
        if (len(X_train_new)) in maxlabel2:
                rf.fit(X_train_new, Y_train_new)
                g_preds = rf.predict(X_test)
                g_MSE[K] += sqrt(sum(1/X_test.shape[0]*(Y_test - g_preds)**2))
                error_g[K][tree] = g_MSE[K]
                K+=1
    
    np.savetxt("supcond_graph.csv", error_g, delimiter=",")

#Consists of code for QBC using regression trees and EMCM using Gradient Boosting Trees

import numpy as np
import math
from math import sqrt
from sklearn.tree import DecisionTreeRegressor 
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import preprocessing,datasets, linear_model
import copy
import scipy
import pandas as pd
from collections import Counter, defaultdict
from sklearn.cluster import KMeans
from scipy.cluster.vq import vq
from math import dist
from numpy.linalg import norm
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import BaggingRegressor
from sklearn.linear_model import LinearRegression

feat=81   
maxtree=200   #total runs
maxlabel1 = 15           #first n_init
maxlabel4 = [maxlabel1]
maxlabel2=[20,40,60,80,100,120,140,160,180,200,220,240,260,280,300,320,340,360,380,400]
maxlabel3 = [20-maxlabel1,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20]

error_qbc=np.zeros((len(maxlabel2),maxtree))
error_emcm=np.zeros((len(maxlabel2),maxtree))

min_samples_leaf = 3     #min samples in a leaf in RF predictor
treeseed = None

rf = RandomForestRegressor(min_samples_leaf = min_samples_leaf)

df = pd.read_csv('data_sets/supercond.csv')
X = df.iloc[:, :-1]
Y = df.iloc[:, -1]

n_qbc = 5   #number of models in the committee
n_emcm = 4   #number of models for EMCM
numberTrees = 25    #number of trees in GBT for EMCM, set to 25 for the superconductivity dataset, but to the default value of 100 for other datasets

for tree in range(maxtree):
    print('Tree_',tree)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.75,random_state=tree)

    X_train1 = X_train    
    Y_train1 = Y_train
    X_train1 = X_train1.to_numpy()    
    Y_train1 = Y_train1.to_numpy()
    X_train2 = X_train1                 
    
    scaler = preprocessing.StandardScaler().fit(X_train1)   
    X_train1 = scaler.transform(X_train1)

    treeseed = None

    qbc_MSE = np.zeros([len(maxlabel2)])
    emcm_MSE = np.zeros([len(maxlabel2)])
    
    n,p = X_train.shape  
                
    cv_ind = np.random.permutation(range(X_train2.shape[0]))
    X_train3 = X_train1[cv_ind,:]        
    Y_train3 = Y_train1[cv_ind]
    X_train2 = X_train2[cv_ind,:]     
    
    ########################################## QBC  ################################################
    
    X_train_new=[]
    Y_train_new=[]
    
    X_train_new=X_train2[:maxlabel1]
    Y_train_new=Y_train3[:maxlabel1]
    X_train2 = np.delete(X_train2,range(maxlabel1),axis = 0)
    X_train3 = np.delete(X_train3,range(maxlabel1),axis = 0)
    Y_train3 = np.delete(Y_train3,range(maxlabel1))

    K=0        
    for rep in range(maxlabel1,maxlabel2[len(maxlabel2)-1]):
        n_samples = int(len(X_train_new) - np.round(len(X_train_new)/n_qbc))
        model = []
        for i in range(n_qbc):
            
            bootstrap = np.random.permutation(range(X_train_new.shape[0]))
            
            X_train_temp = X_train_new[bootstrap,:]
            Y_train_temp = Y_train_new[bootstrap]
            X_train_boot = X_train_temp[:n_samples]
            Y_train_boot = Y_train_temp[:n_samples]
            
            model = np.append(model, (DecisionTreeRegressor().fit(X_train_boot,Y_train_boot))) 
            
        var = []
        for j in range(len(X_train2)):
            pred = [model[i].predict(X_train2[j].reshape(1, -1)) for i in range(len(model))]
            var = np.append(var, np.var(pred))
        label = np.argmax(var)
        
        X_train_new = np.array(np.append(X_train_new,X_train2[label]))
        X_train_new = np.array(np.hsplit(X_train_new,len(X_train_new)/int(feat)))
        Y_train_new = np.append(Y_train_new,Y_train3[label])
        X_train2 = np.delete(X_train2,label,axis = 0)
        Y_train3 = np.delete(Y_train3,label)
        
        if (len(X_train_new)) in maxlabel2:
                rf.fit(X_train_new, Y_train_new)
                g_preds = rf.predict(X_test)
                qbc_MSE[K] += sqrt(sum(1/X_test.shape[0]*(Y_test - g_preds)**2))
                error_qbc[K][tree] = qbc_MSE[K]
                K+=1
     
    np.savetxt("supercond_qbc.csv", error_qbc, delimiter=",")
    
    
    ###################################### EMCM ############################################
    
    X_train1 = X_train    
    Y_train1 = Y_train
    X_train1 = X_train1.to_numpy()    
    Y_train1 = Y_train1.to_numpy()
    X_train2 = X_train1                 
    
    scaler = preprocessing.StandardScaler().fit(X_train1)   
    X_train1 = scaler.transform(X_train1)
    
    cv_ind = np.random.permutation(range(X_train2.shape[0]))
    X_train3 = X_train1[cv_ind,:]       
    Y_train3 = Y_train1[cv_ind]
    X_train2 = X_train2[cv_ind,:]      
    
    X_train_new=[]
    Y_train_new=[]
    
    #get initial labeled set via random sampling
    
    X_train_new=X_train2[:maxlabel1]
    Y_train_new=Y_train3[:maxlabel1]
    X_train2 = np.delete(X_train2,range(maxlabel1),axis = 0)
    X_train3 = np.delete(X_train3,range(maxlabel1),axis = 0)
    Y_train3 = np.delete(Y_train3,range(maxlabel1))
    
    first_model = GradientBoostingRegressor().fit(X_train_new,Y_train_new)
    f_x = first_model.predict(X_train2)

    K=0        
    for rep in range(maxlabel1,maxlabel2[len(maxlabel2)-1]):
        n_samples = int(len(X_train_new) - np.round(len(X_train_new)/n_emcm))   #samples for bootstrapping
        
        #create n_emcm models via bootstrap
        model = []
        for i in range(n_emcm):
            
            bootstrap = np.random.permutation(range(X_train_new.shape[0]))
            
            X_train_temp = X_train_new[bootstrap,:]
            Y_train_temp = Y_train_new[bootstrap]
            X_train_boot = X_train_temp[:n_samples]
            Y_train_boot = Y_train_temp[:n_samples]
            
            model = np.append(model, (GradientBoostingRegressor().fit(X_train_boot,Y_train_boot)))
            
        model_change = []
        for j in range(len(X_train2)):
            #generate super features for non-linear case
            phi = [first_model.estimators_.flatten()[k].predict(X_train2[j].reshape(1, -1)) for k in range(numberTrees)]
                
            temp = [norm((f_x[j] - model[i].predict(X_train2[j].reshape(1, -1)))*phi) for i in range(len(model))]
            model_change = np.append(model_change, np.sum(temp)/int(n_emcm))
        label = np.argmax(model_change)
        
        X_train_new = np.array(np.append(X_train_new,X_train2[label]))
        X_train_new = np.array(np.hsplit(X_train_new,len(X_train_new)/int(feat)))
        Y_train_new = np.append(Y_train_new,Y_train3[label])
        X_train2 = np.delete(X_train2,label,axis = 0)
        Y_train3 = np.delete(Y_train3,label)
        
        first_model = GradientBoostingRegressor().fit(X_train_new,Y_train_new)
        f_x = first_model.predict(X_train2)
        
        if (len(X_train_new)) in maxlabel2:
                rf.fit(X_train_new, Y_train_new)
                g_preds = rf.predict(X_test)
                emcm_MSE[K] += sqrt(sum(1/X_test.shape[0]*(Y_test - g_preds)**2))
                error_emcm[K][tree] = emcm_MSE[K]
                K+=1
                
        np.savetxt("supercond_emcm.csv", error_emcm, delimiter=",")
   


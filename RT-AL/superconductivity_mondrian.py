###########################  Mondrian tree algorithm, sequentially done, labeling 20 samples at each step

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
import random
from math import dist
from mondrian import Mondrian_Tree

import warnings
warnings.filterwarnings('ignore')

maxtree=200        #total number of runs/experiments
maxlabel1 = 15     #number of first n_init samples 
maxlabel4 = [maxlabel1]
maxlabel2=[20,40,60,80,100,120,140,160,180,200,220,240,260,280,300,320,340,360,380,400]  #set of total number of samples to be labeled
maxlabel3 = [20-maxlabel1,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20]   #set of number of samples to be labeled after every step

error_mondrian=np.zeros((len(maxlabel2),maxtree))

feat=81                #number of input features 
min_samples_leaf = 3   #min_samples_in_leaf for the random forest predictor
treeseed = None
tree_seed = None

rf = RandomForestRegressor(min_samples_leaf = min_samples_leaf)

df = pd.read_csv('data_sets/supercond.csv')
X = df.iloc[:, :-1]
Y = df.iloc[:, -1]

def scale_zero_one(col):

        offset = min(col)
        scale = max(col) - min(col)
        col = (col - offset)/scale
        return(col)
        
for tree in range(maxtree):

    print('Tree_',tree)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.75,random_state=tree)

    X_train1 = X_train    ]
    Y_train1 = Y_train
    X_train1 = X_train1.to_numpy()   
    Y_train1 = Y_train1.to_numpy()
        
    for i in range(X_train1.shape[1]):
        X_train1[:,i] = scale_zero_one(X_train1[:,i])
        
    X_train2 = X_train    
    X_train2 = X_train2.to_numpy()   
        
    treeseed = None

    mondrian_MSE = np.zeros([len(maxlabel2)])
    
    n,p = X_train.shape  
    
    cv_ind = np.random.permutation(range(X_train2.shape[0]))
    
    X_train_new=[]
    Y_train_new=[]
    
    X_train3 = X_train1[cv_ind,:]        #scaled
    Y_train3 = Y_train1[cv_ind]
    X_train2 = X_train2[cv_ind,:]        #unscaled
    
    for i in range(len(maxlabel2)):
    
        X_train_new=X_train2[:int(maxlabel2[i]/2)]
        Y_train_new=Y_train3[:int(maxlabel2[i]/2)]
        
        MT = Mondrian_Tree([[0,1]]*p)
        
        MT.update_life_time((maxlabel2[i]**(1/(2+p))-1), set_seed=tree_seed)
        
        MT.input_data(X_train3, range(int(maxlabel2[i]/2)), Y_train3[:int(maxlabel2[i]/2)])

        MT.make_full_leaf_list()
        MT.make_full_leaf_var_list()
        MT.al_set_default_var_global_var()
            
        MT.al_calculate_leaf_proportions()
        
        MT.al_calculate_leaf_number_new_labels(maxlabel2[i])
        
        new_labelled_points = []
                
        for j, node in enumerate(MT._full_leaf_list):
            curr_num = len(node.labelled_index)
            tot_num = curr_num + MT._al_leaf_number_new_labels[j]
            num_new_points = MT._al_leaf_number_new_labels[j]
            labels_to_add = node.pick_new_points(num_new_points,self_update = False, set_seed = tree_seed)
            new_labelled_points.extend(labels_to_add)
            for ind in labels_to_add:
                MT.label_point(ind, Y_train3[ind])
                Y_train_new = np.append(Y_train_new,Y_train3[ind])
                X_train_new = np.append(X_train_new,X_train2[ind])
        X_train_new = np.hsplit(X_train_new,len(X_train_new)/int(feat))
        MT.set_default_pred_global_mean()

        rf.fit(X_train_new,Y_train_new)
        MT_preds = rf.predict(X_test)
        mondrian_MSE[i] += sqrt(sum(1/X_test.shape[0]*(Y_test - MT_preds)**2))
        error_mondrian[i][tree] = mondrian_MSE[i]
        
        np.savetxt("super_mondrian.csv", error_mondrian, delimiter=",")

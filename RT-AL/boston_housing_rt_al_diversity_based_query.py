#Consists python codes for Random sampling, GSx and RT-AL with diversity based query criteria

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
import warnings
warnings.filterwarnings('ignore')

feat=13    # input features
cmax=5
maxtree=200              #total runs
maxlabel1 = 15           #first n_init sampels
maxlabel4 = [maxlabel1]
maxlabel2=[20,40,60,80,100,120,140,160,180,200]    #set of samples to be labeled
maxlabel3 = [20-maxlabel1,20,20,20,20,20,20,20,20,20]

error_rn=np.zeros((len(maxlabel2),maxtree))
error_gsx=np.zeros((len(maxlabel2),maxtree))
error_gsx_bt_diversity=np.zeros((len(maxlabel2),maxtree))

min_samples_leaf = 3     #min samples in a leaf in RF predictor
treeseed = None

rf = RandomForestRegressor(min_samples_leaf = min_samples_leaf)
ridge2 = Ridge(alpha=0.1)     #for gsy

df = pd.read_csv('data_sets/housing.csv')
X = df.iloc[:, :-1]
Y = df.iloc[:, -1]

#for cube method, using the R file Cube.R
import rpy2
from rpy2 import *
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
randomForest = importr('randomForest')
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter
import random

def ClusterIndices(clustNum, labels_array): 
            return np.where(labels_array == clustNum)[0]
    
from collections import Counter, defaultdict

from sklearn.cluster import KMeans
from scipy.cluster.vq import vq
from math import dist

from tree import Breiman_Tree
from tree_diversity import Breiman_Tree_diversity
from tree_representativity import Breiman_Tree_representativity

for tree in range(maxtree):
    print('Tree_',tree)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.75,random_state=tree)
    
    with localconverter(ro.default_converter + pandas2ri.converter):
      X_train_r = ro.conversion.py2rpy(X_train)

    X_train1 = X_train    
    Y_train1 = Y_train
    X_train1 = X_train1.to_numpy()    
    Y_train1 = Y_train1.to_numpy()
    X_train2 = X_train1                  
    
    scaler = preprocessing.StandardScaler().fit(X_train1)    #scaled for distance calculations
    X_train1 = scaler.transform(X_train1)

    treeseed = None

    rn_MSE = np.zeros([len(maxlabel2)])
    gsx_MSE = np.zeros([len(maxlabel2)])
    gsx_bt_diversity_MSE = np.zeros([len(maxlabel2)])
    
    n,p = X_train.shape  
        
    #############################################  Random Sampling (RS)   ######################################
    
    cv_ind = np.random.permutation(range(X_train2.shape[0]))
    X_train_uni = X_train2[cv_ind,:]
    Y_train_uni = Y_train1[cv_ind]
    
    for i,j in enumerate(maxlabel2):
        rf.fit(X_train_uni[list(range(j)),:], Y_train_uni[list(range(j))])
        rn_preds = rf.predict(X_test)
        rn_MSE[i] += sqrt(sum(1/X_test.shape[0]*(Y_test - rn_preds)**2))
        error_rn[i][tree] = rn_MSE[i]
        np.savetxt("housing_rs.csv", error_rn delimiter=",")
    
        
    #############################################  GSx   ######################################
    
    dist_mat = scipy.spatial.distance_matrix(X_train3, X_train3)   #distance matrix of train set
    z=np.zeros(len(dist_mat))
    for i in range(len(dist_mat)):
        z[i]=sum(dist_mat[i])
    centroid = np.argmin(z)   #index of centroid
    X_train_new=[]
    Y_train_new=[]
    X_train_new = np.array(np.append(X_train_new,X_train2[centroid]))
    Y_train_new = np.append(Y_train_new,Y_train3[centroid])
    
    D2 = dist_mat[centroid]
    X_train_new = np.array(np.append(X_train_new,X_train2[np.argmax(D2)]))
    Y_train_new = np.append(Y_train_new,Y_train3[np.argmax(D2)])

    all_ind = [i for i in range(0,len(dist_mat))]   
    labeled_ind = [centroid,np.argmax(D2)]
    D2 = np.delete(D2,centroid)
    K=0    
    for rep in range(2,maxlabel2[len(maxlabel2)-1]):
        unlab_ind=np.setdiff1d(all_ind,labeled_ind)
        D2 = np.delete(D2,np.argmax(D2))   #remove last labeled point from D2
        D=[]
        D = [np.append(D,dist_mat[labeled_ind[len(labeled_ind)-1]][i]) for i in unlab_ind]  #dist of unlabto last lab
        D3 = np.append(D2,D)
        D3=D3.reshape(2,-1)
        D2 = [D3[:,i].min() for i in range(len(D3[0]))]
        new_ind = unlab_ind[np.argmax(D2)]
        labeled_ind = np.append(labeled_ind,new_ind)
        X_train_new = np.array(np.append(X_train_new,X_train2[new_ind])) 
        Y_train_new = np.array(np.append(Y_train_new,Y_train3[new_ind])) 
        X_train_new = np.hsplit(X_train_new,len(X_train_new)/int(feat))
        if (len(X_train_new)) in maxlabel2:
                rf.fit(X_train_new, Y_train_new)
                gsx_preds = rf.predict(X_test)
                gsx_MSE[K] += sqrt(sum(1/X_test.shape[0]*(Y_test - gsx_preds)**2))
                error_gsx[K][tree] = gsx_MSE[K]
                K+=1
        np.savetxt("housing_gsx.csv", error_gsx, delimiter=",")
                
    X_train_gsx = X_train_new
    Y_train_gsx = Y_train_new
    labeled_ind_gsx = labeled_ind

            
   ######################################### GSx + RT-AL(Diversity-based query)  ###########################################
    
    X_train_new=[]
    Y_train_new=[]
    
    X_train_new=X_train_gsx[:maxlabel1]
    Y_train_new=Y_train_gsx[:maxlabel1]
    
    BT10 = Breiman_Tree_diversity(seed=treeseed,min_samples_leaf = 5)
    BT10.input_data(X_train3, labeled_ind[:maxlabel1], Y_train_gsx[:maxlabel1])
    BT10.fit_tree()
    for i in range(len(maxlabel2)):
        BT10.al_calculate_leaf_proportions()
        new_points = BT10.pick_new_points(num_samples = maxlabel3[i])
            
        for new_point in new_points:
            BT10.label_point(new_point, Y_train3[new_point])
            
        for j in new_points:
            Y_train_new = np.append(Y_train_new,Y_train3[j])
            X_train_new = np.append(X_train_new,X_train2[j])
        X_train_new = np.hsplit(X_train_new,len(X_train_new)/int(feat))
        rf.fit(X_train_new,Y_train_new)
        BT_preds = rf.predict(X_test)
        gsx_bt_diversity_MSE[i] += sqrt(sum(1/X_test.shape[0]*(Y_test - BT_preds)**2))
        error_gsx_bt_diversity[i][tree] = gsx_bt_diversity_MSE[i]

        BT10.fit_tree()
        
    np.savetxt("housing_gsx_bt_diversity.csv", error_gsx_bt_diversity, delimiter=",")
   

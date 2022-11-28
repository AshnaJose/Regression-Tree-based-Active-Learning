#Consists python codes for Random sampling, iRDM, RT-AL with representativity based query criteria

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

feat=5   # input features
cmax=5
maxtree=200              #total runs
maxlabel1 = 15           #first n_init sampels
maxlabel4 = [maxlabel1]
maxlabel2=[20,40,60,80,100,120,140,160,180,200]    #set of samples to be labeled
maxlabel3 = [20-maxlabel1,20,20,20,20,20,20,20,20,20]

error_rn=np.zeros((len(maxlabel2),maxtree))
error_irdm=np.zeros((len(maxlabel2),maxtree))
error_irdm_bt_irdm=np.zeros((len(maxlabel2),maxtree))

min_samples_leaf = 3     #min samples in a leaf in RF predictor
treeseed = None

rf = RandomForestRegressor(min_samples_leaf = min_samples_leaf)
ridge2 = Ridge(alpha=0.1)     #for gsy

df = pd.read_csv('data_sets/airfoil-self-noise.csv')
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

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.5,random_state=tree)
    
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
    irdm_MSE = np.zeros([len(maxlabel2)])
    irdm_bt_rep_MSE = np.zeros([len(maxlabel2)])
    
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
        np.savetxt("airfoil_rs.csv", error_rn delimiter=",")
    

    
    #############################################  iRDM   ######################################
    
    for i,j in enumerate(maxlabel2):
        centroid=[]
        kmeans = KMeans(n_clusters = j).fit(X_train3)
        centroid = np.append(centroid, kmeans.cluster_centers_)
        centroid = np.hsplit(centroid,len(centroid)/int(feat))
        
        #iRDM
        num_cluster = []
        for num in range (0, len(centroid)):
            num_cluster = np.append(num_cluster,len(ClusterIndices(num, kmeans.labels_)))    #no of points in each cluster
        
        closest_indices = []
        for clust in range(len(centroid)):
            X_train_cluster = []
            for k in range(int(num_cluster[clust])):
                X_train_cluster = np.append(X_train_cluster,X_train3[ClusterIndices(clust, kmeans.labels_)[k]])
                X_train_cluster = np.hsplit(X_train_cluster,len(X_train_cluster)/int(feat))   #list of data points in cluster i
            centroid[clust] = np.reshape(centroid[clust],(1,-1))
            closest, distances = vq(centroid[clust], X_train_cluster)   #closest = indices of X_train_cluster i closest to centroid i
            closest_indices = np.append(closest_indices, ClusterIndices(clust, kmeans.labels_)[closest])
        closest = closest_indices

        new_cent = []
        for s in closest:
            new_cent = np.append(new_cent,X_train3[int(s)])
        new_cent = np.hsplit(new_cent,len(new_cent)/int(feat))

        X_train_new = []
        Y_train_new = []
        
        rx = []
        for p in range(maxlabel2[i]):
            data_clust = [X_train3[ClusterIndices(p, kmeans.labels_)[q]] for q in range(len(ClusterIndices(p, kmeans.labels_)))]                
            dist_mat_clust = scipy.spatial.distance_matrix(data_clust, data_clust) 
            rxx = [sum(dist_mat_clust[q])/(num_cluster[p]-1) for q in range(len(ClusterIndices(p, kmeans.labels_)))]
            rx.append(rxx)
        
        c2 = []
        for u in range (cmax):
            c1=[]
            for p in range(maxlabel2[i]):
                d = []
                for n in range(len(ClusterIndices(p, kmeans.labels_))):
                    d_gsx = []
                    for l,m in enumerate(closest):
                        if p!=l:
                            d_gsx = np.append(d_gsx,dist(new_cent[l],X_train3[ClusterIndices(p, kmeans.labels_)[n]]))
                    d_gsx = np.delete(d_gsx, np.where(d_gsx==0))
                    d = np.append(d,np.amin(d_gsx))
                c= np.argmax(d-rx[p])
                c1 = np.append(c1,c)
                new_cent[p] = X_train3[ClusterIndices(p, kmeans.labels_)[c]]
            c2.append(c1)

            if u>0 and np.all(c2[u] == c2[u-1]) or u==cmax-1:
                c2 = np.array(c2)
                c2 = c2.astype(int)
                
                for c2_idx, irdm_idx in enumerate(c2[u]):
                    X_train_new = np.append(X_train_new,X_train2[ClusterIndices(c2_idx, kmeans.labels_)[irdm_idx]])
                    Y_train_new = np.append(Y_train_new,Y_train3[ClusterIndices(c2_idx, kmeans.labels_)[irdm_idx]])
                break
                
        X_train_new = np.array(X_train_new)
        Y_train_new = np.array(Y_train_new)
        X_train_new = np.hsplit(X_train_new,len(X_train_new)/int(feat))
        
        rf.fit(X_train_new, Y_train_new)
        irdm_preds = rf.predict(X_test)
        irdm_MSE[i] += sqrt(sum(1/X_test.shape[0]*(Y_test - irdm_preds)**2))
        error_irdm[i][tree] = irdm_MSE[i]
        np.savetxt("airfoil_irdm.csv", error_irdm, delimiter=",")
            
    #############################################  iRDM + RT-AL (Representativity-based query)   ######################################
    
    for i,j in enumerate(maxlabel4):
        centroid=[]
        kmeans = KMeans(n_clusters = j).fit(X_train3)
        centroid = np.append(centroid, kmeans.cluster_centers_)
        centroid = np.hsplit(centroid,len(centroid)/int(feat))
        
        #iRDM
        num_cluster = []
        for num in range (0, len(centroid)):
            num_cluster = np.append(num_cluster,len(ClusterIndices(num, kmeans.labels_)))    #no of points in each cluster
        
        closest_indices = []
        for clust in range(len(centroid)):
            X_train_cluster = []
            for k in range(int(num_cluster[clust])):
                X_train_cluster = np.append(X_train_cluster,X_train3[ClusterIndices(clust, kmeans.labels_)[k]])
                X_train_cluster = np.hsplit(X_train_cluster,len(X_train_cluster)/int(feat))   #list of data points in cluster i
            centroid[clust] = np.reshape(centroid[clust],(1,-1))
            closest, distances = vq(centroid[clust], X_train_cluster)   #closest = indices of X_train_cluster i closest to centroid i
            closest_indices = np.append(closest_indices, ClusterIndices(clust, kmeans.labels_)[closest])
        closest = closest_indices

        new_cent = []
        for s in closest:
            new_cent = np.append(new_cent,X_train3[int(s)])
        new_cent = np.hsplit(new_cent,len(new_cent)/int(feat))

        X_train_new = []
        Y_train_new = []
        
        rx = []
        for p in range(maxlabel4[i]):
            data_clust = [X_train3[ClusterIndices(p, kmeans.labels_)[q]] for q in range(len(ClusterIndices(p, kmeans.labels_)))]                
            dist_mat_clust = scipy.spatial.distance_matrix(data_clust, data_clust) 
            rxx = [sum(dist_mat_clust[q])/(num_cluster[p]-1) for q in range(len(ClusterIndices(p, kmeans.labels_)))]
            rx.append(rxx)
        
        c2 = []
        for u in range (cmax):
            c1=[]
            for p in range(maxlabel4[i]):
                d = []
                for n in range(len(ClusterIndices(p, kmeans.labels_))):
                    d_gsx = []
                    for l,m in enumerate(closest):
                        if p!=l:
                            d_gsx = np.append(d_gsx,dist(new_cent[l],X_train3[ClusterIndices(p, kmeans.labels_)[n]]))
                    d_gsx = np.delete(d_gsx, np.where(d_gsx==0))
                    d = np.append(d,np.amin(d_gsx))
                c= np.argmax(d-rx[p])
                c1 = np.append(c1,c)
                new_cent[p] = X_train3[ClusterIndices(p, kmeans.labels_)[c]]
            c2.append(c1)

            if u>0 and np.all(c2[u] == c2[u-1]) or u==cmax-1:
                c2 = np.array(c2)
                c2 = c2.astype(int)
                labeled_ind = c2[u]
                
                for c2_idx, irdm_idx in enumerate(c2[u]):
                    X_train_new = np.append(X_train_new,X_train2[ClusterIndices(c2_idx, kmeans.labels_)[irdm_idx]])
                    Y_train_new = np.append(Y_train_new,Y_train3[ClusterIndices(c2_idx, kmeans.labels_)[irdm_idx]])
                break
                
        X_train_new = np.array(X_train_new)
        Y_train_new = np.array(Y_train_new)
        X_train_new = np.hsplit(X_train_new,len(X_train_new)/int(feat))
        
    X_train_irdm = X_train_new
    Y_train_irdm = Y_train_new
            
    BT11 = Breiman_Tree_representativity(seed=treeseed,min_samples_leaf = 5)
    BT11.input_data(X_train3, labeled_ind, Y_train_new)
    BT11.fit_tree()
    for i in range(len(maxlabel2)):
        BT11.al_calculate_leaf_proportions()
        new_points = BT11.pick_new_points(num_samples = maxlabel3[i])
            
        for new_point in new_points:
            BT11.label_point(new_point, Y_train3[new_point])
            
        for j in new_points:
            Y_train_new = np.append(Y_train_new,Y_train3[j])
            X_train_new = np.append(X_train_new,X_train2[j])
        X_train_new = np.hsplit(X_train_new,len(X_train_new)/int(feat))
        rf.fit(X_train_new,Y_train_new)
        BT_preds = rf.predict(X_test)
        irdm_bt_rep_MSE[i] += sqrt(sum(1/X_test.shape[0]*(Y_test - BT_preds)**2))
        error_irdm_bt_rep[i][tree] = irdm_bt_rep_MSE[i]

        BT11.fit_tree()
        np.savetxt("airfoil_irdm_bt_rep.csv", error_irdm_bt_rep, delimiter=",")
                

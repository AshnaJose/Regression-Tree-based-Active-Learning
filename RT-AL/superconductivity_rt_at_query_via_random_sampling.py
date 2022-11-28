#Consists pythond codes for Random sampling, GSx, iRDM, Cube, Gsy and RT-AL using random sampling as the query criteria

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

feat=81    # input features
cmax=5
maxtree=200              #total runs
maxlabel1 = 15           #first n_init sampels
maxlabel4 = [maxlabel1]
maxlabel2=[20,40,60,80,100,120,140,160,180,200,220,240,260,280,300,320,340,360,380,400]    #set of samples to be labeled
maxlabel3 = [20-maxlabel1,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20]

error_rn=np.zeros((len(maxlabel2),maxtree))
error_bt=np.zeros((len(maxlabel2),maxtree))
error_gsx=np.zeros((len(maxlabel2),maxtree))
error_gsx_bt_diversity=np.zeros((len(maxlabel2),maxtree))
error_cube=np.zeros((len(maxlabel2),maxtree))
error_irdm=np.zeros((len(maxlabel2),maxtree))
error_irdm_bt_irdm=np.zeros((len(maxlabel2),maxtree))
error_gsy=np.zeros((len(maxlabel2),maxtree))

min_samples_leaf = 3     #min samples in a leaf in RF predictor
treeseed = None

rf = RandomForestRegressor(min_samples_leaf = min_samples_leaf)
ridge2 = Ridge(alpha=0.1)     #for gsy

df = pd.read_csv('data_sets/supercond.csv')
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
print(len(X),len(Y))

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
    bt_MSE = np.zeros([len(maxlabel2)])
    gsy_r2_MSE = np.zeros([len(maxlabel2)])
    gsx_MSE = np.zeros([len(maxlabel2)])
    gsx_bt_diversity_MSE = np.zeros([len(maxlabel2)])
    cube_MSE = np.zeros([len(maxlabel2)])
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
        np.savetxt("supercond_rs.csv", error_rn delimiter=",")
    
    #################################################### RT-AL (query criteria in leaves - random sampling) ################################################
    
    X_train_new=[]
    Y_train_new=[]
    
    X_train3 = X_train1[cv_ind,:]        
    Y_train3 = Y_train1[cv_ind]
    X_train2 = X_train2[cv_ind,:]        
    X_train_new=X_train2[:maxlabel1]
    Y_train_new=Y_train3[:maxlabel1]
    BT = Breiman_Tree(seed=treeseed,min_samples_leaf = 5)
    BT.input_data(X_train3, range(maxlabel1), Y_train3[:maxlabel1])
    BT.fit_tree()
    for i in range(len(maxlabel2)):
        BT.al_calculate_leaf_proportions()
        new_points = BT.pick_new_points(num_samples = maxlabel3[i])
            
        for new_point in new_points:
            BT.label_point(new_point, Y_train3[new_point])
            
        for j in new_points:
            Y_train_new = np.append(Y_train_new,Y_train3[j])
            X_train_new = np.append(X_train_new,X_train2[j])
        X_train_new = np.hsplit(X_train_new,len(X_train_new)/int(feat))
        rf.fit(X_train_new,Y_train_new)
        BT_preds = rf.predict(X_test)
        bt_MSE[i] += sqrt(sum(1/X_test.shape[0]*(Y_test - BT_preds)**2))
        error_bt[i][tree] = bt_MSE[i]

        BT.fit_tree()
        np.savetxt("supercond_bt.csv", error_bt, delimiter=",")
        
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
        np.savetxt("supercond_gsx.csv", error_gsx, delimiter=",")
                
    X_train_gsx = X_train_new
    Y_train_gsx = Y_train_new
    labeled_ind_gsx = labeled_ind
                    
    
    #################################################  Cube  #############################################
    
    r = robjects.r
    robjects.r('''rm(list=ls()) 
    library(BalancedSampling)
    library(FactoMineR)
    library(rpart)
    library(treeClust)
    library(randomForest)
    library(mclust)
    library(FactoMineR)
    ''')
    r['source']('Cube.R')
    first_sampling_r = robjects.globalenv['first_sampling']
    
    for i,j in enumerate(maxlabel2):
        X_train_new=[]
        Y_train_new = []
        label=[]
        iter=0
        if len(label)!=j and iter<100:
            label = first_sampling_r(X_train_r,j,'cube')
            iter+=1
        label = np.array(label)
        label-=1
            
        if len(label)!=j:
            remain = len(label) - j
            if remain>0:
                for m in range(remain):
                    label = np.delete(label,random.randint(0, len(label)-1))
                    
            elif remain<0:
                all_ind = np.array([r for r in range(0,len(X_train1))])
                for n in range(-1*remain):
                    label = np.append(label,random.choice(np.setdiff1d(all_ind,label)))
        
        for k in label:
            X_train_new = np.append(X_train_new,X_train2[k])
            Y_train_new = np.append(Y_train_new,Y_train3[k])
        X_train_new = np.hsplit(X_train_new,len(X_train_new)/int(feat))
        rf.fit(X_train_new, Y_train_new)
        cube_preds = rf.predict(X_test)
        cube_MSE[i] += sqrt(sum(1/X_test.shape[0]*(Y_test - cube_preds)**2))
        error_cube[i][tree] = cube_MSE[i]
        np.savetxt("supercond_cube.csv", error_cube, delimiter=",")
    
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
        np.savetxt("supercond_irdm.csv", error_irdm, delimiter=",")
                
    ######################################################### GSy  ###############################################            
                
                
    X_train1 = X_train    
    Y_train1 = Y_train
    X_train1 = X_train1.to_numpy()   
    Y_train1 = Y_train1.to_numpy()
    X_train2 = X_train1                  
    
    scaler = preprocessing.StandardScaler().fit(X_train1)   
    X_train1 = scaler.transform(X_train1)
    
    X_train3 = X_train1[cv_ind,:]       
    Y_train3 = Y_train1[cv_ind]
    X_train2 = X_train2[cv_ind,:]       

    X_train_new=X_train_gsx[:maxlabel1]
    Y_train_new=Y_train_gsx[:maxlabel1]
    
    labeled_ind =labeled_ind_gsx[:maxlabel1]
    
    labeled_ind=np.sort(labeled_ind)
    for i in range(len(labeled_ind)-1,-1,-1):
        X_train3 = np.delete(X_train3,i,axis=0)
        Y_train3 = np.delete(Y_train3,i,axis=0)
        X_train2 = np.delete(X_train2,i,axis=0)
            
    ridge2.fit(X_train_new, Y_train_new)     
    clf_preds = ridge2.predict(X_train2)
    
    dyn=np.zeros(len(clf_preds))
    j=0
    for f in clf_preds:
        dyn[j]=np.amin([abs(f-g) for g in Y_train_new])
        j+=1
    dyn_idx=np.argmax(dyn)
    X_train_new = np.array(np.append(X_train_new,X_train2[dyn_idx]))   
    Y_train_new = np.append(Y_train_new,Y_train3[dyn_idx])
    X_train_new = np.hsplit(X_train_new,len(X_train_new)/int(feat))
    X_train2 = np.delete(X_train2,dyn_idx,axis = 0)
    Y_train3 = np.delete(Y_train3,dyn_idx)

    ridge2.fit(X_train_new, Y_train_new)            
    clf_preds = ridge2.predict(X_train2)
    dyn=np.zeros(len(clf_preds))
    j=0
    for f in clf_preds:
        dyn[j]=np.amin([abs(f-g) for g in Y_train_new])
        j+=1
    D2 = dyn

    K=0   
    for rep in range(maxlabel1+1,maxlabel2[len(maxlabel2)-1]):
        D=[abs(clf_preds[g]-Y_train_new[len(Y_train_new)-1]) for g in range(len(clf_preds))]
        D3 = np.append(D2,D)
        D3=D3.reshape(2,-1)
        D2 = [D3[:,i].min() for i in range(len(D3[0]))]
        new_ind = np.argmax(D2)
        X_train_new = np.array(np.append(X_train_new,X_train2[new_ind])) 
        Y_train_new = np.array(np.append(Y_train_new,Y_train3[new_ind])) 
        X_train_new = np.hsplit(X_train_new,len(X_train_new)/int(feat))
        X_train2 = np.delete(X_train2,new_ind,axis = 0)
        Y_train3 = np.delete(Y_train3,new_ind)
        D2 = np.delete(D2,np.argmax(D2))        
        ridge2.fit(X_train_new, Y_train_new)
        clf_preds = ridge2.predict(X_train2)
        dyn=np.zeros(len(clf_preds))
        j=0
        for f in clf_preds:
            dyn[j]=np.amin([abs(f-g) for g in Y_train_new])
            j+=1
        D2 = dyn

        if (len(X_train_new)) in maxlabel2:
                rf.fit(X_train_new, Y_train_new)
                gsy_preds = rf.predict(X_test)
                gsy_r2_MSE[K] += sqrt(sum(1/X_test.shape[0]*(Y_test - gsy_preds)**2))
                error_gsy_r2[K][tree] = gsy_r2_MSE[K]
                K+=1
                
        np.savetxt("supercond_gsy.csv", error_gsy, delimiter=",")

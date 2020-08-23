#!/usr/bin/env python3

"""
============================================================
Comparison of clustering algorithms with CluReAL refinement
(multi-dimensional datasets)
============================================================

FIV, Jun 2020

"""

import time
import warnings

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from sklearn import cluster, datasets, mixture
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
from itertools import cycle, islice

import sys
import clureal as cr

np.random.seed(0)

jset  = sys.argv[1]
alg  = sys.argv[2]
verb  = (sys.argv[3]=='1')

num_datasets = 20
folder = "dataMd/"
sets = {'separated','low-noise','high-noise','dens-diff','close','complex'}
algs = {'bench','mkm','gmm','ahc'}

if verb:
    print(__doc__)
    print("Data folder: ", folder)
    print("Dataset set: ", jset)
    print("Number of datasets: ", num_datasets)
    print("Algorithm: ", alg)

warnings.filterwarnings('ignore')

params = {'n_neighbors': 10,'n_clusters': 15}

if (alg in algs and jset in sets):
    for i in range(1,num_datasets+1): 
        filename = folder+jset+'_'+str(i)#+'.csv'
        dataset = np.genfromtxt(filename, delimiter=',')

        X, y = dataset[:,0:-1], dataset[:,-1].astype(int)

        # normalize dataset
        X = StandardScaler().fit_transform(X)

        # connectivity matrix for structured Ward
        connectivity = kneighbors_graph(X, n_neighbors=params['n_neighbors'], include_self=False)
        # make connectivity symmetric
        connectivity = 0.5 * (connectivity + connectivity.T)

        if alg=="mkm":
            algorithm = cluster.MiniBatchKMeans(n_clusters=params['n_clusters'])
        elif alg=="ahc":
            algorithm = cluster.AgglomerativeClustering(linkage="average", affinity="cityblock",n_clusters=params['n_clusters'], connectivity=connectivity)
        elif alg=="gmm":
            algorithm = mixture.GaussianMixture(n_components=params['n_clusters'], covariance_type='full')

        if alg=="bench":
            t0 = time.time()
            _, dataset, _ = cr.clureal_complete(X,y,refinement=False, SK=False, report=False)
            m,n = X.shape
            k = np.max(y)
            out = np.sum(y==-1)/m
            S = dataset['Silhouette']
            CH = dataset['Calinski Harabasz']/1000
            DB = dataset['Davies Bouldin'] 
            Grex,Gstr,Gmin = dataset['Grex'],dataset['Gstr'],dataset['Gmin'] 
            t1 = time.time()
            print("Set: %s, Dataset: %d, Algorithm: %s, m: %d, n:%d, k:%d, out:%.2f, Gstr: %.2f, Grex: %.2f, Gmin: %.2f, Sil: %.2f, C-H: %.2f, D-B: %.2f, time: %.2f" % ( jset, i, alg, m, n, k, out, Gstr, Grex, Gmin, S, CH, DB, t1-t0))

        else:
            t0 = time.time()
            # catch warnings related to kneighbors_graph
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="the number of connected components of the " +
                    "connectivity matrix is [0-9]{1,2}" +
                    " > 1. Completing it to avoid stopping the tree early.",
                    category=UserWarning)
                warnings.filterwarnings(
                    "ignore",
                    message="Graph is not fully connected, spectral embedding" +
                    " may not work as expected.",
                    category=UserWarning)
                algorithm.fit(X)

            t1 = time.time()
            if hasattr(algorithm, 'labels_'):
                y_pred = algorithm.labels_.astype(np.int)
            else:
                y_pred = algorithm.predict(X)

            t0 = time.time()
            _, dataset, _ = cr.clureal_complete(X,y_pred,refinement=False, SK=False, report=False)
            t1 = time.time()
            m,n = X.shape
            k = np.max(y_pred)
            out = np.sum(y_pred==-1)/m
            S = dataset['Silhouette']
            CH = dataset['Calinski Harabasz']/1000
            DB = dataset['Davies Bouldin'] 
            Grex,Gstr,Gmin = dataset['Grex'],dataset['Gstr'],dataset['Gmin'] 
            print("Set: %s, Dataset: %d, Algorithm: %s, m: %d, n:%d, k:%d, out:%.2f, Gstr: %.2f, Grex: %.2f, Gmin: %.2f, Sil: %.2f, C-H: %.2f, D-B: %.2f, time: %.2f" % ( jset, i, alg, m, n, k, out, Gstr, Grex, Gmin, S, CH, DB, t1-t0))

            t0 = time.time()
            y_pred, dataset, _ = cr.clureal_complete(X,y_pred,refinement=True, SK=False, report=False, repetitions=2)
            t1 = time.time()
            m,n = X.shape
            k = np.max(y_pred)
            out = np.sum(y_pred==-1)/m
            S = dataset['Silhouette']
            CH = dataset['Calinski Harabasz']/1000
            DB = dataset['Davies Bouldin'] 
            Grex,Gstr,Gmin = dataset['Grex'],dataset['Gstr'],dataset['Gmin'] 
            print("Set: %s, Dataset: %d, Algorithm: %s+CR, m: %d, n:%d, k:%d, out:%.2f, Gstr: %.2f, Grex: %.2f, Gmin: %.2f, Sil: %.2f, C-H: %.2f, D-B: %.2f, time: %.2f" % ( jset, i, alg, m, n, k, out, Gstr, Grex, Gmin, S, CH, DB, t1-t0))

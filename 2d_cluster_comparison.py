#!/usr/bin/env python3

"""
============================================================
Comparison of clustering algorithms with CluReAL refinement 
(2D datasets)
============================================================

FIV, Jun 2020

"""
print(__doc__)

import time
import warnings

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from sklearn import cluster, datasets, mixture
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
from itertools import cycle, islice

import clureal as cr

np.random.seed(0)

from numpy import genfromtxt
sepd = genfromtxt('data2d/separated_15.csv', delimiter=',')
lout = genfromtxt('data2d/low-noise_19.csv', delimiter=',')
hout = genfromtxt('data2d/high-noise_2.csv', delimiter=',')
dens = genfromtxt('data2d/dens-diff_19.csv', delimiter=',')
clos = genfromtxt('data2d/close_18.csv', delimiter=',')
cplx = genfromtxt('data2d/complex_9.csv', delimiter=',')
data_names = ['separated','low-out','high-out','dens-diff','close','complex']

plt.figure(figsize=(9 * 2 + 3, 12.5))
plt.subplots_adjust(left=.02, right=.98, bottom=.001, top=.96, wspace=.05,hspace=.01)
gridspec.GridSpec(12,18)

default_base = {'n_neighbors': 10,
                'n_clusters': 15,}

datasets = [
    (sepd, {}),
    (lout, {}),
    (hout, {}),
    (dens, {}),
    (clos, {}),
    (cplx, {})]

for i_dataset, (dataset, algo_params) in enumerate(datasets):
    # update parameters with dataset-specific values
    params = default_base.copy()
    params.update(algo_params)

    X, y = dataset[:,0:2], dataset[:,2].astype(int)

    # normalize dataset
    X = StandardScaler().fit_transform(X)

    # connectivity matrix for structured Ward
    connectivity = kneighbors_graph(X, n_neighbors=params['n_neighbors'], include_self=False)
    # make connectivity symmetric
    connectivity = 0.5 * (connectivity + connectivity.T)

    mkm = cluster.MiniBatchKMeans(n_clusters=params['n_clusters'])
    ahc = cluster.AgglomerativeClustering(linkage="average", affinity="cityblock",n_clusters=params['n_clusters'], connectivity=connectivity)
    gmm = mixture.GaussianMixture(n_components=params['n_clusters'], covariance_type='full')

    clustering_algorithms = (('MKM', mkm),('AHC', ahc),('GMM', gmm))

    alg=-1
    for name, algorithm in clustering_algorithms:
        alg +=1
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

        print("\n******************************************************", i_dataset, name)

        plt.figure(1)

        plt.subplot2grid((12,18), (2*i_dataset,alg*6), colspan=2, rowspan=2)

        if i_dataset == 0:
            plt.title(name, size=16)

        if alg==0:
            plt.ylabel(data_names[i_dataset], size=16)
        
        colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a',
                                             '#f781bf', '#a65628', '#984ea3',
                                             '#999999', '#e41a1c', '#dede00','#266da7', '#ee6e00', '#3cae39',
                                             '#e670ae', '#954517', '#873d92',
                                             '#888888', '#d3090b', '#cdcd00']),int(max(y_pred) + 1))))
        # add black color for outliers (if any)
        colors = np.append(colors, ["#000000"])

        plt.scatter(X[:, 0], X[:, 1], s=1, color=colors[y_pred])
        plt.xlim(np.min(X[:,0])-0.1, np.max(X[:,0])+0.1)
        plt.ylim(np.min(X[:,1])-0.1, np.max(X[:,1]+0.1))
        plt.xticks(())
        plt.yticks(())

        plt.subplot2grid((12,18), (2*i_dataset,alg*6+2))
        _, dataset, _ = cr.clureal_complete(X,y_pred,refinement=False, SK=True, report=True)

        #plt.show()

        plt.subplot2grid((12,18), (2*i_dataset+1,alg*6+2))
        S = dataset['Silhouette']
        CH = dataset['Calinski Harabasz']/1000
        DB = dataset['Davies Bouldin'] 
        Grex,Gstr,Gmin = dataset['Grex'],dataset['Gstr'],dataset['Gmin'] 
        plt.text(0, 1, ('Gr: %.2f' % Grex).lstrip('0'),transform=plt.gca().transAxes, size=10, horizontalalignment='left')
        plt.text(0, .8, ('Gs: %.2f' % Gstr).lstrip('0'),transform=plt.gca().transAxes, size=10, horizontalalignment='left')
        plt.text(0, .6, ('Gm: %.2f' % Gmin).lstrip('0'),transform=plt.gca().transAxes, size=10, horizontalalignment='left')
        plt.text(0, .4, ('S: %.2f' % S).lstrip('0'),transform=plt.gca().transAxes, size=10, horizontalalignment='left')
        plt.text(0, .2, ('CH: %.2fM' % CH).lstrip('0'),transform=plt.gca().transAxes, size=10, horizontalalignment='left')
        plt.text(0, 0, ('DB: %.2f' % DB).lstrip('0'),transform=plt.gca().transAxes, size=10, horizontalalignment='left')
        plt.axis('off')

        plt.subplot2grid((12,18), (2*i_dataset,alg*6+5))
        y_pred, dataset, _ = cr.clureal_complete(X,y_pred,refinement=True,SK=True, report=True, repetitions=2)

        plt.subplot2grid((12,18), (2*i_dataset+1,alg*6+5))
        S = dataset['Silhouette']
        CH = dataset['Calinski Harabasz']/1000
        DB = dataset['Davies Bouldin'] 
        Grex,Gstr,Gmin = dataset['Grex'],dataset['Gstr'],dataset['Gmin'] 
        plt.text(0, 1, ('Gr: %.2f' % Grex).lstrip('0'),transform=plt.gca().transAxes, size=10, horizontalalignment='left')
        plt.text(0, .8, ('Gs: %.2f' % Gstr).lstrip('0'),transform=plt.gca().transAxes, size=10, horizontalalignment='left')
        plt.text(0, .6, ('Gm: %.2f' % Gmin).lstrip('0'),transform=plt.gca().transAxes, size=10, horizontalalignment='left')
        plt.text(0, .4, ('S: %.2f' % S).lstrip('0'),transform=plt.gca().transAxes, size=10, horizontalalignment='left')
        plt.text(0, .2, ('CH: %.2fM' % CH).lstrip('0'),transform=plt.gca().transAxes, size=10, horizontalalignment='left')
        plt.text(0, 0, ('DB: %.2f' % DB).lstrip('0'),transform=plt.gca().transAxes, size=10, horizontalalignment='left')
        plt.axis('off')

        plt.subplot2grid((12,18), (2*i_dataset,alg*6+3), colspan=2, rowspan=2)

        if i_dataset == 0:
            plt.title(name+'+ CluReAL', size=16)

        colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a',
                                             '#f781bf', '#a65628', '#984ea3',
                                             '#999999', '#e41a1c', '#dede00','#266da7', '#ee6e00', '#3cae39',
                                             '#e670ae', '#954517', '#873d92',
                                             '#888888', '#d3090b', '#cdcd00']),int(max(y_pred) + 1))))
        # add black color for outliers (if any)
        colors = np.append(colors, ["#000000"])

        plt.scatter(X[:, 0], X[:, 1], s=1, color=colors[y_pred])
        plt.xlim(np.min(X[:,0])-0.1, np.max(X[:,0])+0.1)
        plt.ylim(np.min(X[:,1])-0.1, np.max(X[:,1]+0.1))
        plt.xticks(())
        plt.yticks(())

plt.show()

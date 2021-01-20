"""
==============================================
Comparison of clustering optimization methods 
with 2d-data and k-means algorithm
 
FIV, Jan 2021
==============================================
"""

#!/usr/bin/env python3

print(__doc__)

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib as matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd

from sklearn import cluster, mixture
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.cluster import adjusted_mutual_info_score

import clureal as cr

np.random.seed(100)

def get_cmap(n, name='hsv'):
    return plt.cm.get_cmap(name, n)

data_names = ["a2", "a3", "close", "complex", "dens-diff", "high-noise", "low-noise", "s1", "s2", "s3","separated","unbalance"]
 
plt.figure(1,figsize=(21, 12.5), dpi=80)
#plt.figure(1,figsize=(12, 10), dpi=80)
plt.subplots_adjust(left=.02, right=.98, bottom=.001, top=.96, wspace=.05,hspace=.01)
gridspec.GridSpec(12,18)

for d_ind, d_name in enumerate(data_names):

    file_name = "data2d/"+d_name+".csv"
    dataset = np.genfromtxt(file_name, delimiter=',')

    print("\n------- DATASET: ", d_name, "-------")

    X, ygt = dataset[:,0:2], dataset[:,2].astype(int)

    p_outliers = False
    if min(ygt)==-1:
        p_outliers = True   
    p_n_clusters = max(ygt)

    # normalize dataset
    X = StandardScaler().fit_transform(X)

    range_k = np.arange(10)-5+p_n_clusters
    if min(range_k)<2:
        range_k+=2-min(range_k)

    print("Silhouette Sweep")

    perf = np.ones(len(range_k))
    for i in range_k:
        algorithm = cluster.MiniBatchKMeans(n_clusters=i, random_state=100)
        y = algorithm.fit_predict(X)
        s,_,_ = cr.other_validations(X,y)
        perf[i-min(range_k)] = s

    best_k = np.argmax(perf)+min(range_k)      
    algorithm = cluster.MiniBatchKMeans(n_clusters=best_k, random_state=100)
    y = algorithm.fit_predict(X)

    cc = cr.cluster_context(X,y)
    gv = cr.gval(cc)

    if p_outliers==False:
        y = cr.reassign_outliers(X,y,0,cc.centroids,gv.extR).astype(int)

    S,CH,DB = cr.other_validations(X,y)
    rc = cr.refinement_context(X,y,cc,gv)
    AMI = adjusted_mutual_info_score(ygt, y)

    print('- Grex:', round(gv.Grex,2), ', Gstr:', round(gv.Gstr,2), ', Gmin:', round(gv.Gmin,2))
    print('- Sil:', round(S,2), ', CH:', round(CH,2), ', DB:', round(DB,2))
    print('- AMI:', round(AMI,2))

    plt.subplot(2, 2, 1)
    cmap = get_cmap(max(y+2),'tab20b')
    plt.scatter(X[:, 0], X[:, 1], s=2, color=cmap(y+1))
    plt.scatter(X[y==-1, 0], X[y==-1, 1], s=2, c='k')
    plt.title("Clustering (best Silhouette sweep)")
    plt.subplot(2, 2, 3)
    cc = cr.cluster_context(X,y)
    gv = cr.gval(cc)
    cr.draw_symbol(cc, gv, rc)

    print("CluReAL")

    k = 10 + p_n_clusters
    algorithm = cluster.MiniBatchKMeans(n_clusters=k, random_state=100)
    y = algorithm.fit_predict(X)

    cc = cr.cluster_context(X,y)
    gv = cr.gval(cc)
    rc = cr.refinement_context(X,y,cc,gv)

    if p_outliers:
        y,cc = cr.refine(X,y,cc,gv,rc)
    else:
        y,cc = cr.refine(X,y,cc,gv,rc,0, min_rdens = -0.9, min_mass = 0.001, out_sens = 0 )


    gv = cr.gval(cc)
    S,CH,DB = cr.other_validations(X,y)
    AMI = adjusted_mutual_info_score(ygt, y)

    print('- Grex:', round(gv.Grex,2), ', Gstr:', round(gv.Gstr,2), ', Gmin:', round(gv.Gmin,2))
    print('- Sil:', round(S,2), ', CH:', round(CH,2), ', DB:', round(DB,2))
    print('- AMI:', round(AMI,2))

    rc = cr.refinement_context(X,y,cc,gv)

    plt.subplot(2, 2, 2)
    cmap = get_cmap(max(y+2),'tab20b')
    plt.scatter(X[:, 0], X[:, 1], s=2, color=cmap(y+1))
    plt.scatter(X[y==-1, 0], X[y==-1, 1], s=2, c='k')
    plt.title("Clustering with CluReAL refinement")
    plt.subplot(2, 2, 4)
    cc = cr.cluster_context(X,y)
    gv = cr.gval(cc)
    cr.draw_symbol(cc, gv, rc)

    nameout = "plots/k_"+d_name+".png"
    plt.tight_layout() 
    plt.savefig(nameout, format='png')

    plt.clf()



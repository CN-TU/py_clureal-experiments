"""
==============================================
Study of CRAL options when facing high 
cluster overlap
 
FIV, May 2021
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

data_names = ["s2", "s3"]
 
for d_ind, d_name in enumerate(data_names):

    plt.figure(1,figsize=(20, 8), dpi=80)
    plt.subplots_adjust(left=.02, right=.98, bottom=.001, top=.96, wspace=.05,hspace=.01)

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

    print("Ground Truth")

    plt.subplot(2, 5, 1)
    cmap = get_cmap(max(ygt+3),'tab20b')
    plt.scatter(X[:, 0], X[:, 1], s=2, color=cmap(ygt+1))
    plt.scatter(X[ygt==-1, 0], X[ygt==-1, 1], s=2, c='k')
    plt.title("Ground Truth")
    plt.subplot(2, 5, 6)
    cc = cr.cluster_context(X,ygt)
    gv = cr.gval(cc)
    rc = cr.refinement_context(X,ygt,cc,gv)
    cr.draw_symbol(cc, gv, rc)

    print("CluReAL")

    k = 10 + p_n_clusters
    algorithm = cluster.MiniBatchKMeans(n_clusters=k, random_state=100)
    y_base = algorithm.fit_predict(X)

    CR_config_options = [(False, 0), (False, 1), (True, 0), (True, 1)]

    for idx, (coreset,prun_level) in enumerate(CR_config_options):

        y = y_base
        print("Config., (coreset, pruning level): ",coreset,prun_level)
        if coreset == True:
            Xo,yo = X,y
            X,y,ind = cr.coreset_extractor(Xo,yo,k=int(0.7*len(Xo)))

        cc = cr.cluster_context(X,y)
        gv = cr.gval(cc)
        rc = cr.refinement_context(X,y,cc,gv)

        if p_outliers:
            y,cc = cr.refine(X,y,cc,gv,rc, prun_level = prun_level)
        else:
            y,cc = cr.refine(X,y,cc,gv,rc,0, min_rdens = -0.9, min_mass = 0.001, out_sens = 0, prun_level = prun_level )

        if coreset == True:
            gv = cr.gval(cc)
            rc = cr.refinement_context(X,y,cc,gv)
            yn = np.ones(len(yo))*-1
            yn[ind] = y.astype(int)
            y = cr.reassign_outliers(Xo,yn,0,cc.centroids,gv.extR).astype(int)
            X = Xo

        gv = cr.gval(cc)
        S,CH,DB = cr.other_validations(X,y)
        AMI = adjusted_mutual_info_score(ygt, y)

        print('- Grex:', round(gv.Grex,2), ', Gstr:', round(gv.Gstr,2), ', Gmin:', round(gv.Gmin,2))
        print('- Sil:', round(S,2), ', CH:', round(CH,2), ', DB:', round(DB,2))
        print('- AMI:', round(AMI,2))

        rc = cr.refinement_context(X,y,cc,gv)

        plt.subplot(2, 5, 2+idx)
        cmap = get_cmap(max(y+3),'tab20b')
        plt.scatter(X[:, 0], X[:, 1], s=2, color=cmap(y+1))
        plt.scatter(X[y==-1, 0], X[y==-1, 1], s=2, c='k')
        title_text = "CRAL: coreset="+str(coreset)+" , prun.lev.="+str(prun_level)
        plt.title(title_text)
        plt.subplot(2, 5, 7+idx)
        cc = cr.cluster_context(X,y)
        gv = cr.gval(cc)
        cr.draw_symbol(cc, gv, rc)

    nameout = "plots/ho_"+d_name+".png"
    plt.tight_layout() 
    plt.savefig(nameout, format='png')

    plt.close()



"""
==============================================
Sensitivity analysis of dataset size
Comparison of runtimes of clustering optimization 
methods, partitional algorithms (k-dependent)
 
FIV, May 2021
==============================================
"""

#!/usr/bin/env python3

print(__doc__)

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import time

from sklearn import cluster, mixture
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.cluster import adjusted_mutual_info_score
from sklearn.datasets import make_blobs

import clureal as cr


np.random.seed(100)

def get_cmap(n, name='hsv'):
    return plt.cm.get_cmap(name, n)

def mkm(k):
    model = cluster.MiniBatchKMeans(n_clusters=k, random_state=100)
    return model
 
def ahc(k):
    model = cluster.AgglomerativeClustering(linkage="average", affinity="cityblock",n_clusters=k)
    return model
 
def gmm(k):
    model = mixture.GaussianMixture(n_components=k, covariance_type='full',random_state=100)
    return model

def bir(k):
    model = cluster.Birch(n_clusters=k)
    return model 

def select_algorithm(argument,k):
    switcher = {'mkm': mkm, 'ahc': ahc, 'gmm': gmm, 'bir': bir}
    model = switcher.get(argument, lambda: "Invalid algorithm")
    return model(k)

### LOG file
data_sizes = [500,1000,2500,5000,10000,25000]
algs = [ "ahc", "bir", "gmm", "mkm"]
methods = ['Best', 'CRAL']
df_columns = ["Grex", "Gstr", "Gmin", "Sil", "CH", "DB","AMI","Time"]

iterables = [data_sizes,algs,methods]
df_index = pd.MultiIndex.from_product(iterables, names=['Data', 'Alg.','Method'])
df_val = pd.DataFrame(columns=df_columns,index=df_index)

### DATASETS

for d_ind, num_datapoints in enumerate(data_sizes):

    centers = 30 #np.random.random_integers(20,50)
    n_features = 5 #np.random.random_integers(5,10)
    X, ygt = make_blobs(n_samples=num_datapoints, centers=centers, n_features=n_features, random_state=0)

    print("\n------- DATASET: ", str(num_datapoints), "-------")
    print("centers",centers)
    print("n_features",n_features)

    # normalize dataset
    X = StandardScaler().fit_transform(X)

    ### CLUSTERING ALGORITHMS 
    range_k = np.arange(20)+20

    clustering_algorithms = ['ahc','bir','gmm','mkm']

    for a_name in clustering_algorithms:

        print("Silhouette Sweep:", a_name)

        perf = np.ones(len(range_k))
        start = time.time()
        for i in range_k:
            algorithm = select_algorithm(a_name,i)
            y = algorithm.fit_predict(X)
            s,_,_ = cr.other_validations(X,y)
            perf[i-min(range_k)] = s

        best_k = np.argmax(perf)+min(range_k)        
        algorithm = select_algorithm(a_name,best_k)
        y = algorithm.fit_predict(X)
        end = time.time()
        T = (end - start)

        cc = cr.cluster_context(X,y)
        gv = cr.gval(cc)

        p_outliers = False
        if p_outliers==False:
            y = cr.reassign_outliers(X,y,0,cc.centroids,gv.extR).astype(int)

        S,CH,DB = cr.other_validations(X,y)
        rc = cr.refinement_context(X,y,cc,gv)
        AMI = adjusted_mutual_info_score(ygt, y)

        df_val.loc[(num_datapoints,a_name,'Best'), 'Grex'] = gv.Grex
        df_val.loc[(num_datapoints,a_name,'Best'), 'Gstr'] = gv.Gstr
        df_val.loc[(num_datapoints,a_name,'Best'), 'Gmin'] = gv.Gmin
        df_val.loc[(num_datapoints,a_name,'Best'), 'Sil'] = S
        df_val.loc[(num_datapoints,a_name,'Best'), 'CH'] = CH
        df_val.loc[(num_datapoints,a_name,'Best'), 'DB'] = DB
        df_val.loc[(num_datapoints,a_name,'Best'), 'AMI'] = AMI
        df_val.loc[(num_datapoints,a_name,'Best'), 'Time'] = T
        print("AMI, Time:", AMI, T)

        print("CluReAL:", a_name)
        k = 50 
        algorithm = select_algorithm(a_name,k)
        start = time.time()
        y = algorithm.fit_predict(X)

        cc = cr.cluster_context(X,y)
        gv = cr.gval(cc)
        rc = cr.refinement_context(X,y,cc,gv)
        if p_outliers:
            y,cc = cr.refine(X,y,cc,gv,rc,0)
        else:
            y,cc = cr.refine(X,y,cc,gv,rc,0, min_rdens = -0.9, min_mass = 0.001, out_sens = 0 )

        end = time.time()
        T = (end - start)
        gv = cr.gval(cc)
        S,CH,DB = cr.other_validations(X,y)
        AMI = adjusted_mutual_info_score(ygt, y)

        df_val.loc[(num_datapoints,a_name,'CRAL'), 'Grex'] = gv.Grex
        df_val.loc[(num_datapoints,a_name,'CRAL'), 'Gstr'] = gv.Gstr
        df_val.loc[(num_datapoints,a_name,'CRAL'), 'Gmin'] = gv.Gmin
        df_val.loc[(num_datapoints,a_name,'CRAL'), 'Sil'] = S
        df_val.loc[(num_datapoints,a_name,'CRAL'), 'CH'] = CH
        df_val.loc[(num_datapoints,a_name,'CRAL'), 'DB'] = DB
        df_val.loc[(num_datapoints,a_name,'CRAL'), 'AMI'] = AMI
        df_val.loc[(num_datapoints,a_name,'CRAL'), 'Time'] = T
        print("AMI, Time:", AMI, T)

        rc = cr.refinement_context(X,y,cc,gv)

df_val.to_csv('results/runtime_results_complete_k.csv')

#out_table = df_sum.to_latex(caption="MultiD-experiments results")
#text_file = open('results/runtime_results_k.tex', "w")
#text_file.write(out_table)
#text_file.close()



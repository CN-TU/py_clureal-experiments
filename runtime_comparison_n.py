"""
==============================================
Sensitivity analysis of dataset size
Comparison of runtimes of clustering optimization 
methods, density-based algorithms
 
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
from sklearn.neighbors import NearestNeighbors
import hdbscan
from kneed import KneeLocator

import clureal as cr


np.random.seed(100)

def get_cmap(n, name='hsv'):
    return plt.cm.get_cmap(name, n)

def hdbs(m,e,s,x):
    model = hdbscan.HDBSCAN(min_cluster_size = s, min_samples = int(m), cluster_selection_epsilon = float(e))
    return model
 
def opt(m,e,s,x):
    model = cluster.OPTICS(min_samples = int(m), xi = x, min_cluster_size = s)
    return model

def select_algorithm(argument,m,e,s,x):
    switcher = {'hdbs': hdbs, 'opt': opt}
    model = switcher.get(argument, lambda: "Invalid algorithm")
    return model(m,e,s,x)

### LOG file
data_sizes = [500,1000,2500,5000,10000,25000]
algs = [ "hdbs", "opt"]
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
    dim_m, dim_n = X.shape

    print("\n------- DATASET: ", str(num_datapoints), "-------")
    print("centers",centers)
    print("n_features",n_features)

    # normalize dataset
    X = StandardScaler().fit_transform(X)

    ### CLUSTERING ALGORITHMS 
    range_k = np.arange(20)+20

    clustering_algorithms = ['hdbs','opt']

    for a_name in clustering_algorithms:

        print("Best clustering (with parameter search):", a_name)

        n_combinations = 20
        max_dim = 3*dim_n 
        if max_dim> 100:
            max_dim = 100

        start = time.time()
        m = np.around(np.linspace(5, max_dim, num=n_combinations))
        neighbors = NearestNeighbors(n_neighbors=2)
        neighbors_fit = neighbors.fit(X)
        distances, indices = neighbors_fit.kneighbors(X)
        distances = distances[:,1]
        distances = np.sort(distances, axis=0).flatten()
        kn = KneeLocator(np.arange(len(distances)), distances, curve='convex', direction='increasing')
        kni = kn.knee
        e = np.linspace(distances[kni]/2, 2*distances[kni], num=n_combinations)
        x = 0.05 + 0.15*np.random.random_sample((n_combinations,))

        perf = np.zeros(n_combinations)
        for i in range(n_combinations):
            algorithm = select_algorithm(a_name,m[i],e[i],int(dim_m*0.05),x[i])
            y = algorithm.fit_predict(X)
            if (sum(y+1)==0):
                y[:]=0
            s,_,_ = cr.other_validations(X,y)
            perf[i] = s

        if sum(np.isnan(perf)==False)==0:
            best_e = 10*max(e)
            best_x = 0.1
            best_m = 5
        else:
            best_e = e[np.nanargmax(perf)]     
            best_x = x[np.nanargmax(perf)]     
            best_m = m[np.nanargmax(perf)]

        algorithm = select_algorithm(a_name,best_m,best_e,int(dim_m*0.05),best_x)
        y = algorithm.fit_predict(X)

        cc = cr.cluster_context(X,y)
        gv = cr.gval(cc)

        p_outliers = False
        if p_outliers==False:
            y = cr.reassign_outliers(X,y,0,cc.centroids,gv.extR).astype(int)

        end = time.time()
        T = (end - start)
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

        def_m = 5
        def_e = distances[kni]
        def_x = 0.05
        algorithm = select_algorithm(a_name,def_m,def_e,int(dim_m*0.05),def_x)
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

df_val.to_csv('results/runtime_results_complete_n.csv')

#out_table = df_sum.to_latex(caption="MultiD-experiments results")
#text_file = open('results/runtime_results_n.tex', "w")
#text_file.write(out_table)
#text_file.close()



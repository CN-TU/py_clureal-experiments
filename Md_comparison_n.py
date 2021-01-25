"""
==============================================
Comparison of clustering optimization methods 
with multi-dimensional data for density-based
algorithms
 
FIV, Jan 2021
==============================================
"""

#!/usr/bin/env python3

print(__doc__)

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn import cluster
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.cluster import adjusted_mutual_info_score
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
data_names = ['close_1', 'close_10', 'close_11', 'close_12', 'close_13', 'close_14', 'close_15', 'close_16', 'close_17', 'close_18', 'close_19', 'close_2', 'close_20', 'close_3', 'close_4', 'close_5', 'close_6', 'close_7', 'close_8', 'close_9', 'complex_1', 'complex_10', 'complex_11', 'complex_12', 'complex_13', 'complex_14', 'complex_15', 'complex_16', 'complex_17', 'complex_18', 'complex_19', 'complex_2', 'complex_20', 'complex_3', 'complex_4', 'complex_5', 'complex_6', 'complex_7', 'complex_8', 'complex_9', 'dens-diff_1', 'dens-diff_10', 'dens-diff_11', 'dens-diff_12', 'dens-diff_13', 'dens-diff_14', 'dens-diff_15', 'dens-diff_16', 'dens-diff_17', 'dens-diff_18', 'dens-diff_19', 'dens-diff_2', 'dens-diff_20', 'dens-diff_3', 'dens-diff_4', 'dens-diff_5', 'dens-diff_6', 'dens-diff_7', 'dens-diff_8', 'dens-diff_9', 'high-noise_1', 'high-noise_10', 'high-noise_11', 'high-noise_12', 'high-noise_13', 'high-noise_14', 'high-noise_15', 'high-noise_16', 'high-noise_17', 'high-noise_18', 'high-noise_19', 'high-noise_2', 'high-noise_20', 'high-noise_3', 'high-noise_4', 'high-noise_5', 'high-noise_6', 'high-noise_7', 'high-noise_8', 'high-noise_9', 'low-noise_1', 'low-noise_10', 'low-noise_11', 'low-noise_12', 'low-noise_13', 'low-noise_14', 'low-noise_15', 'low-noise_16', 'low-noise_17', 'low-noise_18', 'low-noise_19', 'low-noise_2', 'low-noise_20', 'low-noise_3', 'low-noise_4', 'low-noise_5', 'low-noise_6', 'low-noise_7', 'low-noise_8', 'low-noise_9', 'multidim_0002', 'multidim_0003', 'multidim_0005', 'multidim_0010', 'multidim_0015', 'multidim_0032', 'multidim_0064', 'multidim_0256', 'multidim_0512', 'multidim_1024', 'separated_1', 'separated_10', 'separated_11', 'separated_12', 'separated_13', 'separated_14', 'separated_15', 'separated_16', 'separated_17', 'separated_18', 'separated_19', 'separated_2', 'separated_20', 'separated_3', 'separated_4', 'separated_5', 'separated_6', 'separated_7', 'separated_8', 'separated_9']
sets_name = ['close','complex','dens-diff','high-noise','low-noise','multidim','separated']
algs = [ "hdbs", "opt"]
methods = ['Best', 'CRAL']
df_columns = ["Grex", "Gstr", "Gmin", "Sil", "CH", "DB","AMI","T"]

iterables = [data_names,algs,methods]
df_index = pd.MultiIndex.from_product(iterables, names=['Data', 'Alg.','Method'])
df_val = pd.DataFrame(columns=df_columns,index=df_index)

iterables = [sets_name,algs,methods]
df_index = pd.MultiIndex.from_product(iterables, names=['Data', 'Alg.','Method'])
df_sum = pd.DataFrame(columns=df_columns,index=df_index)

### DATASETS

for d_ind, d_name in enumerate(data_names):

    file_name = "dataMd/"+d_name
    dataset = np.genfromtxt(file_name, delimiter=',')
    print("\n------- DATASET: ", d_name, "-------")

    X, ygt = dataset[:,0:-2], dataset[:,-1].astype(int)

    dim_m, dim_n = X.shape
    p_outliers = False
    if min(ygt)==-1:
        p_outliers = True   
    p_n_clusters = max(ygt)

    # normalize dataset
    X = StandardScaler().fit_transform(X)

    ### CLUSTERING ALGORITHMS 

    clustering_algorithms = ['hdbs','opt']

    for a_name in clustering_algorithms:

        print("Best clustering (with parameter search):", a_name)

        n_combinations = 20
        max_dim = 3*dim_n 
        if max_dim> 100:
            max_dim = 100

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

        if p_outliers==False:
            y = cr.reassign_outliers(X,y,0,cc.centroids,gv.extR).astype(int)

        S,CH,DB = cr.other_validations(X,y)
        rc = cr.refinement_context(X,y,cc,gv)
        AMI = adjusted_mutual_info_score(ygt, y)

        df_val.loc[(d_name,a_name,'Best'), 'Grex'] = gv.Grex
        df_val.loc[(d_name,a_name,'Best'), 'Gstr'] = gv.Gstr
        df_val.loc[(d_name,a_name,'Best'), 'Gmin'] = gv.Gmin
        df_val.loc[(d_name,a_name,'Best'), 'Sil'] = S
        df_val.loc[(d_name,a_name,'Best'), 'CH'] = CH
        df_val.loc[(d_name,a_name,'Best'), 'DB'] = DB
        df_val.loc[(d_name,a_name,'Best'), 'AMI'] = AMI

        print("CluReAL:", a_name)

        def_m = 5
        def_e = distances[kni]
        def_x = 0.1
        algorithm = select_algorithm(a_name,def_m,def_e,int(dim_m*0.05),def_x)
        y = algorithm.fit_predict(X)

        cc = cr.cluster_context(X,y)
        gv = cr.gval(cc)
        rc = cr.refinement_context(X,y,cc,gv)

        if p_outliers:
            y,cc = cr.refine(X,y,cc,gv,rc,0)
        else:
            y,cc = cr.refine(X,y,cc,gv,rc,0, min_rdens = -0.9, min_mass = 0.001, out_sens = 0 )

        gv = cr.gval(cc)
        S,CH,DB = cr.other_validations(X,y)
        AMI = adjusted_mutual_info_score(ygt, y)

        df_val.loc[(d_name,a_name,'CRAL'), 'Grex'] = gv.Grex
        df_val.loc[(d_name,a_name,'CRAL'), 'Gstr'] = gv.Gstr
        df_val.loc[(d_name,a_name,'CRAL'), 'Gmin'] = gv.Gmin
        df_val.loc[(d_name,a_name,'CRAL'), 'Sil'] = S
        df_val.loc[(d_name,a_name,'CRAL'), 'CH'] = CH
        df_val.loc[(d_name,a_name,'CRAL'), 'DB'] = DB
        df_val.loc[(d_name,a_name,'CRAL'), 'AMI'] = AMI

        rc = cr.refinement_context(X,y,cc,gv)

for setj in sets_name:
    df_aux = df_val.iloc[df_val.index.get_level_values(0).str.contains(setj)]
    for a_name in clustering_algorithms:
        df_aux2 = df_aux.iloc[df_aux.index.get_level_values(1).str.contains(a_name)]
        df_auxB = df_aux2.iloc[df_aux2.index.get_level_values(2).str.contains('Best')]
        df_auxC = df_aux2.iloc[df_aux2.index.get_level_values(2).str.contains('CRAL')]
        df_sum.loc[(setj,a_name,'CRAL')] = df_auxC.mean()
        df_sum.loc[(setj,a_name,'Best')] = df_auxB.mean()

df_val.to_csv('results/n_Md_results_complete.csv')
df_sum.to_csv('results/n_Md_results_sum.csv')

#out_table = df_sum.to_latex(caption="MultiD-experiments results")
#text_file = open('results/n_Md_results_sum.tex', "w")
#text_file.write(out_table)
#text_file.close()



"""
==============================================
Comparison of CluReAL v1 vs CluReAL v2 
with k-means algorithm
 
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

import clureal as cr2
import clureal_v1 as cr1

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
data_names = ['real_4', 'close_2', 'complex_3', 'dens-diff_4', 'high-noise_5', 'low-noise_6', 'multidim_0032', 'separated_7']
sets_name = ['real','close','complex','dens-diff','high-noise','low-noise','multidim','separated']
algs = [ "mkm"]
methods = ['Normal', 'CRALv1', 'CRALv2']
df_columns = ["Grex", "Gstr", "Gmin", "Sil", "CH", "DB","AMI","Time"]

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

    X, ygt = dataset[:,0:-1], dataset[:,-1].astype(int)

    p_outliers = False
    if min(ygt)==-1:
        p_outliers = True   
    p_n_clusters = max(ygt)

    # normalize dataset
    X = StandardScaler().fit_transform(X)

    ### CLUSTERING ALGORITHMS 
    clustering_algorithms = ['mkm']

    for a_name in clustering_algorithms:

        print("Clustering with wrong parameters:", a_name)

        start = time.time()

        k = 10 + p_n_clusters
        algorithm = select_algorithm(a_name,k)
        y = algorithm.fit_predict(X)
        end = time.time()
        T = (end - start)

        cc = cr2.cluster_context(X,y)
        gv = cr2.gval(cc)

        if p_outliers==False:
            y = cr2.reassign_outliers(X,y,0,cc.centroids,gv.extR).astype(int)

        S,CH,DB = cr2.other_validations(X,y)
        rc = cr2.refinement_context(X,y,cc,gv)
        AMI = adjusted_mutual_info_score(ygt, y)

        df_val.loc[(d_name,a_name,'Normal'), 'Grex'] = gv.Grex
        df_val.loc[(d_name,a_name,'Normal'), 'Gstr'] = gv.Gstr
        df_val.loc[(d_name,a_name,'Normal'), 'Gmin'] = gv.Gmin
        df_val.loc[(d_name,a_name,'Normal'), 'Sil'] = S
        df_val.loc[(d_name,a_name,'Normal'), 'CH'] = CH
        df_val.loc[(d_name,a_name,'Normal'), 'DB'] = DB
        df_val.loc[(d_name,a_name,'Normal'), 'AMI'] = AMI
        df_val.loc[(d_name,a_name,'Normal'), 'Time'] = T
        print(AMI,T)

        print("CluReAL v2:", a_name)
        k = 10 + p_n_clusters
        algorithm = select_algorithm(a_name,k)
        start = time.time()
        y = algorithm.fit_predict(X)

        cc = cr2.cluster_context(X,y)
        gv = cr2.gval(cc)
        rc = cr2.refinement_context(X,y,cc,gv)
        if p_outliers:
            y,cc = cr2.refine(X,y,cc,gv,rc,0)
        else:
            y,cc = cr2.refine(X,y,cc,gv,rc,0, min_rdens = -0.9, min_mass = 0.001, out_sens = 0 )

        end = time.time()
        T = (end - start)
        gv = cr2.gval(cc)
        S,CH,DB = cr2.other_validations(X,y)
        AMI = adjusted_mutual_info_score(ygt, y)

        df_val.loc[(d_name,a_name,'CRALv2'), 'Grex'] = gv.Grex
        df_val.loc[(d_name,a_name,'CRALv2'), 'Gstr'] = gv.Gstr
        df_val.loc[(d_name,a_name,'CRALv2'), 'Gmin'] = gv.Gmin
        df_val.loc[(d_name,a_name,'CRALv2'), 'Sil'] = S
        df_val.loc[(d_name,a_name,'CRALv2'), 'CH'] = CH
        df_val.loc[(d_name,a_name,'CRALv2'), 'DB'] = DB
        df_val.loc[(d_name,a_name,'CRALv2'), 'AMI'] = AMI
        df_val.loc[(d_name,a_name,'CRALv2'), 'Time'] = T
        print(AMI,T)

        rc = cr2.refinement_context(X,y,cc,gv)

        print("CluReAL v1:", a_name)
        k = 10 + p_n_clusters
        algorithm = select_algorithm(a_name,k)
        start = time.time()
        y = algorithm.fit_predict(X)

        y, dt, _ = cr1.clureal_complete(X,y,refinement=True, SK=False, report=False, repetitions=1)
        end = time.time()
        T = (end - start)

        cc = cr2.cluster_context(X,y)
        gv = cr2.gval(cc)
        S,CH,DB = cr2.other_validations(X,y)
        AMI = adjusted_mutual_info_score(ygt, y)

        df_val.loc[(d_name,a_name,'CRALv1'), 'Grex'] = gv.Grex
        df_val.loc[(d_name,a_name,'CRALv1'), 'Gstr'] = gv.Gstr
        df_val.loc[(d_name,a_name,'CRALv1'), 'Gmin'] = gv.Gmin
        df_val.loc[(d_name,a_name,'CRALv1'), 'Sil'] = S
        df_val.loc[(d_name,a_name,'CRALv1'), 'CH'] = CH
        df_val.loc[(d_name,a_name,'CRALv1'), 'DB'] = DB
        df_val.loc[(d_name,a_name,'CRALv1'), 'AMI'] = AMI
        df_val.loc[(d_name,a_name,'CRALv1'), 'Time'] = T
        print(AMI,T)

for setj in sets_name:
    df_aux = df_val.iloc[df_val.index.get_level_values(0).str.contains(setj)]
    for a_name in clustering_algorithms:
        df_aux2 = df_aux.iloc[df_aux.index.get_level_values(1).str.contains(a_name)]
        df_auxB = df_aux2.iloc[df_aux2.index.get_level_values(2).str.contains('Normal')]
        df_auxC = df_aux2.iloc[df_aux2.index.get_level_values(2).str.contains('CRALv1')]
        df_auxD = df_aux2.iloc[df_aux2.index.get_level_values(2).str.contains('CRALv2')]
        df_sum.loc[(setj,a_name,'CRALv1')] = df_auxC.mean()
        df_sum.loc[(setj,a_name,'CRALv2')] = df_auxD.mean()
        df_sum.loc[(setj,a_name,'Normal')] = df_auxB.mean()

df_val.to_csv('results/v1vs2_CRAL_complete.csv')
df_sum.to_csv('results/v1vs2_CRAL_sum.csv')

#out_table = df_sum.to_latex(caption="MultiD-experiments results")
#text_file = open('results/k_Md_results_sum.tex', "w")
#text_file.write(out_table)
#text_file.close()



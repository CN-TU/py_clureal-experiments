"""
======================
<<<<<<< HEAD
CluReAL algorithm v2.0
FIV, Nov 2020
======================
=======
CluReAL algorithm v1.0
FIV, Jun 2020
======================

>>>>>>> c182d4069ead97e8078f49ef0493d0ff9a1ea816
"""
#!/usr/bin/env python3

import numpy as np
<<<<<<< HEAD
import matplotlib.pyplot as plt
import networkx as nx

from scipy.spatial import distance_matrix
from scipy.spatial import distance
from KDEpy import FFTKDE
from scipy.signal import find_peaks
from itertools import combinations, permutations
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import math
import statistics 

# ********* CluReAL parameters ***********
MIN_REL_DENSITY = -0.8
MIN_CARDINALITY_R = 0.005
OUTLIER_SENS = 0.75
REP = 0
# ***** Symbolic Keys parameters *********
SIMILAR_DENSITY_TH = 3
RADII_RATIO_TH = 2
# ****************************************


class ClusterContext:   #  --- Cluster Context class ---
    def __init__(self,k,m):
        self.k = k  # number of clusters (int)
        self.centroids = np.zeros([k,m])     # matrix (kxm) with robust cluster centers
        self.mass = np.zeros(k)   # cluster mass or cardinality (k-size array)
        self.mnDa = np.zeros(k)   # cluster mean intra distance (k-size array)
        self.mdDa = np.zeros(k)   # cluster median intra distance (k-size array)
        self.sdDa = np.zeros(k)   # cluster std-dev intra distance (k-size array)
        self.De = np.zeros([k,k])     # cluster inter distance matrix (k x k matrix)
        self.outliers = 0         # number of outliers / total data points (float)

class RefinementContext:   #  --- Cluster Refinement Context class ---
    def __init__(self,k):
        self.mm = np.zeros(k)   # multimodality flags for each cluster (k-size array)
        self.kdens = np.zeros(k)   # cluster relative densities (k-size array)
        self.Odens = np.zeros(k)   # global/overall density (float)
        self.kinship = np.zeros([k,k])   # cluster kinship indices (k x k matrix): 4-unrelated, 3-friends, 2-relatives, 1-parent and child, 0-itself

class GValidity: #  --- GValidity class ---
    def __init__(self,k):
        self.Gstr = 0   # strict global index (float)
        self.Grex = 0   # relaxed global index (float)
        self.Gmin = 0   # minimum global index (float)
        self.oi_st = np.zeros(k) # individual strict indices (array of floats)
        self.oi_rx = np.zeros(k) # individual relaxed indices (array of floats)
        self.oi_mn = np.zeros(k) # individual min indices (array of floats)
        self.extR = np.zeros(k)  # extended radii (array of floats)
        self.strR = np.zeros(k)  # strict radii (array of floats)
        self.volR = np.zeros(k)  # times that the extended radious is in the core radious (array of floats)


def sample_size (N, s, e):
	z=1.96
	num = N * pow(z,2) * pow(s,2)
	den = (N-1) * pow(e,2) + pow(z,2) * pow(s,2)
	n = int(math.floor(num/den))
	return n
	
def coreset_extractor(X, y, x=5, qv=0.3, k=None, q=None, chunksize=None):
	
	[m, n] = X.shape
	
	if k is None:
		Xt = StandardScaler().fit_transform(X)
		pca = PCA(n_components=2)
		Xp = pca.fit_transform(Xt)
		sigma = np.std(Xp)
		if sigma<1:
			sigma=1
		error = 0.1*np.std(Xp);
		k = sample_size( m, sigma, error )

	if chunksize is None:
		chunksize = m

	index = np.random.permutation(m)

	O = X[index[0:k]]
	ind = index[0:k]

	P = np.zeros(k)

	for i in range(0,m,chunksize):
		dist = distance.cdist(X[i:(i+chunksize)], O)
		dist_sorted = np.argsort(dist, axis=1)
		closest = dist_sorted[:,0:x].flatten()

		P += np.count_nonzero (closest[:,np.newaxis] == np.arange(k), 0)

	if q is None:
		q = np.quantile(P, qv)

	O = O[P>=q]
	ind = ind[P>=q]

	return O,y[ind],ind


def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w
=======
from scipy.spatial import distance_matrix
from scipy.spatial import distance
from sklearn.neighbors.kde import KernelDensity
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

# ********* CluReAL parameters ***********
REP = 2
DENSE_CLUSTER_TH = 0
HAZY_CLUSTER_TH = -0.5
LOW_MASS_CF = 0.1
SIMILAR_DENSITY_TH = 3
RADII_RATIO_TH = 3
# ****************************************

def rebuilt_labels(y):
    # inputs
    #   y: array with cluster labels (-1 for outliers) 
    # outputs
    #   y_new: refined array with cluster labels (-1 for outliers) 

    y_rem = np.unique(y)
    outs = np.where(y_rem == -1)
    a = 0
    if len(outs[0])>0:
        a=1
    y_new = np.copy(y)
    for i in range(0,y_rem.shape[0]-a):
        y_new[y==y_rem[i+a]]=i
    return y_new

def cluster_refinement(X,y,k,mm,kdens, Odens, kinship, mass, De, volr, oimin):
    # inputs
    #   X: dataset (nxm), matrix of n vectors with m dimensions
    #   y: array with cluster labels (-1 for outliers) 
    #   k: number of clusters (scalar)
    #   mm: multimodality flags for each cluster (kx1-array)
    #   kdens: cluster relative densities (kx1-array)
    #   Odens: global/overall density (scalar)
    #   kinship: cluster kinship indices (k x k matrix): 4-unrelated, 3-friends, 2-relatives, 1-parent and child
    #   mass: cluster mass or cardinality (k-size array)
    #   De: cluster inter distance matrix (k x k matrix)
    #   volr: times that extended radii are in their respective core radii (k-size array)
    #   oimn: oi_min of clusters (k-size array)
    # outputs
    #   y: (refined?) array with cluster labels (-1 for outliers) 

    n, m = X.shape

    for i in range(0,k):
        if mm[i] == 1 and kdens[i]<DENSE_CLUSTER_TH:
            mm[i] = 0

    # fusing clusters based on kinship
    kins = np.where((kinship > 0) & (kinship < 4) )
    ki1,ki2 = kins[0],kins[1]    
    for i in range(0,ki1.shape[0]):
        if (mm[ki1[i]] == 0 and ki1[i]!=ki2[i]):
            Xaux =  np.vstack((X[y==ki1[i],:], X[y==ki2[i],:]))
            dd = np.absolute(kdens[ki1[i]]-kdens[ki2[i]])/np.absolute(np.minimum(kdens[ki1[i]],kdens[ki2[i]]))
            if (multimodality(Xaux)==0): #and dd < 2):
                if dd < SIMILAR_DENSITY_TH:
                    y[y==ki2[i]] = ki1[i]
                    ki1[ki1==ki2[i]] = ki1[i]
                    ki2[ki2==ki2[i]] = ki1[i]
    y = rebuilt_labels(y)
    # dissolving irrelevant clusters
    k,De,mdDa,mnDa,sdDa,mass = extract_cluster_context(X,y)
    Odens,kdens = rdensity (X,y,k)
    min_rho = np.min([2*np.mean(kdens), HAZY_CLUSTER_TH])
    min_mass = LOW_MASS_CF * (np.sum(mass)/k)
    #print(min_rho,min_mass)
    #print(kdens,mass)
    for i in range(0,k):
        if (kdens[i] <= min_rho or mass[i] < min_mass):
            y[y==i] = -1
    y = rebuilt_labels(y)
    return y
>>>>>>> c182d4069ead97e8078f49ef0493d0ff9a1ea816

def create_circle(x,y,r,f,c,v):
    circle= plt.Circle((x,y), radius = r, fill=f, ec='k', fc=c, visible=v)
    return circle

def create_rectangle(x,y,w,h,f,c,v):
    rectangle = plt.Rectangle((x,y), w, h, fill=f, ec='k', fc=c, visible=v)
    return rectangle

def add_shape(patch):
    ax=plt.gca()
    ax.add_patch(patch)
    plt.axis('scaled')

<<<<<<< HEAD
def draw_symbol(cc, gv, rc):
    # inputs
    #   cc: cluster context (ClusterContext)
    #   gv: goi validity indices (GValidty) 
    #   rc: cluster refinement context (RefinementContext)

    k, outliers = cc.k, cc.outliers
    Gstr, Grex, Gmin, volr = gv.Gstr, gv.Grex, gv.Gmin, gv.volR
    mm,kinship,kdens = rc.mm, rc.kinship, rc.kdens

=======
def other_validations(X,y,verbose=False):
    # inputs
    #   X: dataset (nxm), matrix of n vectors with m dimensions
    #   y: array with cluster labels (-1 for outliers) 
    #   verbose: (bool) "True" stands for verbose mode
    # outputs
    #   S: Silhouette index of the whole dataset (scalar) 
    #   CH: Calinski Harabasz index of the whole dataset (scalar) 
    #   DB: Davies Bouldin of the whole dataset (scalar) 

    from sklearn import metrics
    X=X[y!=-1,:]
    y=y[y!=-1]
    S,CH,DB = np.nan, np.nan, np.nan
    k = np.max(y)
    if k>0: 
        S = metrics.silhouette_score(X, y, metric='euclidean')
        CH = metrics.calinski_harabasz_score(X, y)
        DB = metrics.davies_bouldin_score(X, y)
    if verbose:
        print("\nOther metrics:")
        print("Silhouette:", S)
        print("Calinski Harabasz index:", CH)
        print("Davies Bouldin index:", DB)
    return S,CH,DB

def draw_symbol(k,dataset,clusters,mm,kinship,kdens,volr,outliers):
    # inputs
    #   k: number of clusters (scalar)
    #   dataset: (dict) contains validity indices of the whole dataset
    #   clusters: (dict) contains validity indices of individual clusters
    #   mm: multimodality flags for each cluster (kx1-array)
    #   kinship: cluster kinship indices (k x k matrix): 4-unrelated, 3-friends, 2-relatives, 1-parent and child
    #   kdens: cluster relative densities (kx1-array)
    #   volr: times that extended radii are in their respective core radii (k-size array)
    #   outliers: number of outliers (scalar)

    Gstr = dataset['Gstr']
    Grex = dataset['Grex']
    Gmin = dataset['Gmin']
>>>>>>> c182d4069ead97e8078f49ef0493d0ff9a1ea816
    child = np.where(kinship == 1)
    densdiff = np.absolute(np.nanmax(kdens)-np.nanmin(kdens))/np.absolute(np.minimum( np.nanmax(kdens),np.nanmin(kdens) ))  
    volr = np.nanmean(volr)

    x_ec, x_ec2, x_cc, x_cc2, x_ccup, x_ech = 0, 0, 0, 0, 0, 0 
    y_ec, y_cc = 0, 0
    v_ec, v_ec2, v_cc, v_cc2, v_ccup, v_ech, v_r1, v_l1, f_ec2, v_eov = False, False, False, False, False, False, False, False, False, False
    v_ol, v_om, v_oh = False, False, False
    c_ec2 = 'k' 

<<<<<<< HEAD
    if (sum(mm)>0):
=======
    if (np.sum(mm)>0):
>>>>>>> c182d4069ead97e8078f49ef0493d0ff9a1ea816
        y_cc, v_ccup =  -0.08, True
    else:
        y_cc =  0

    if len(child[0])>0:
        v_ech = True

    if Gmin < 0 and k>2 and Gstr>=0 and Grex>=1:
        v_eov = True

    if k==1:
        x_ec, y_ec, v_ec, v_cc =  0, 0, True, True
    elif (k==2 and len(child[0])>0):
        x_ec, y_ec, v_ec, v_cc =  0, 0, True, True
    else:
        if densdiff>=SIMILAR_DENSITY_TH:
            c_ec2, f_ec2 = 'lightgrey', True            
        if Gstr>=1:
            x_ec, x_cc, x_ccup, x_ech, v_ec, v_cc = -0.5, -0.5, -0.5, -0.7, True, True
            x_ec2, x_cc2, v_ec2, v_cc2 = 0.5, 0.5, True, True
        elif Gstr>0:
            if Grex>1:
                x_ec, x_cc, x_ccup, x_ech, v_ec, v_cc = -0.4, -0.4, -0.4, -0.6, True, True
                x_ec2, x_cc2, v_ec2, v_cc2 = 0.4, 0.4, True, True
            else:
                x_ec, x_cc, x_ccup, x_ech, v_ec, v_cc = -0.3, -0.3, -0.3, -0.5, True, True
                x_ec2, x_cc2, v_ec2, v_cc2 = 0.3, 0.3, True, True
        else:
            if Grex>1:
                x_ec, x_cc, x_ccup, x_ech, v_ec, v_cc = -0.15, -0.15, -0.15, -0.35, True, True
                x_ec2, x_cc2, v_ec2, v_cc2 = 0.15, 0.15, True, True
            elif Grex>0:
                v_r1 = True
                x_cc, x_ccup, x_ech, v_cc = -0.2, -0.2, -0.2, True
                x_cc2, v_cc2 = 0.2, True
            else:
                v_l1, v_r1 = True, True
                v_ccup, v_cc, v_cc2, v_ech = False, False, False, False

    if volr<RADII_RATIO_TH:
        v_cc2 = False
        if np.sum(mm)==0:
            v_cc = False

    if outliers>0:
        v_ol = True
        if outliers>0.05:
            v_om = True
            if outliers>0.20:
                v_oh = True

    r1 = create_rectangle(-0.6,-0.4,1.2,0.8,True,'lightgrey',v_r1)
    ec2 = create_circle(x_ec2,0,0.4,f_ec2,c_ec2,v_ec2)
    ec = create_circle(x_ec,y_ec,0.4,False,'k',v_ec)
    ecov = create_circle(x_ec-0.30,0.25,0.12,False,'k',v_eov)
    ech = create_circle(x_ech,-0.20,0.08,False,'k',v_ech)
    cc = create_circle(x_cc,y_cc,0.03,True,'k',v_cc)
    cc2 = create_circle(x_cc2,0,0.03,True,'k',v_cc2)
    ccup = create_circle(x_ccup,0.08,0.03,True,'k',v_ccup)
    o1 = create_circle(-0.15,-0.45,0.02,True,'k',v_om)
    o2 = create_circle(-0.3,-0.50,0.02,True,'k',v_oh)
    o3 = create_circle(0.15,-0.45,0.02,True,'k',v_om)
    o4 = create_circle(0.3,-0.50,0.02,True,'k',v_oh)
    o5 = create_circle(0,-0.50,0.02,True,'k',v_ol)

    if v_l1:
        r1.set_hatch('\\')

    add_shape(r1)
    add_shape(ec2)
    add_shape(ec)
    add_shape(ech)
    add_shape(ecov)
    add_shape(cc)
    add_shape(cc2)
    add_shape(ccup)
    add_shape(o1),add_shape(o2),add_shape(o3),add_shape(o4),add_shape(o5)
    
    ax=plt.gca()
    ax.set_xlim(-1, 1)
    ax.set_ylim(-0.6, 0.6)
    s = str(k)
    plt.text(0, 0.4, s, fontsize=10, ha='center')
    plt.axis('off')

<<<<<<< HEAD
def dig_multimodal(X,y,mm):
    k,c = max(y)+1, -1
    n_clusters = 2
    alg = MiniBatchKMeans(n_clusters=n_clusters, random_state=10)
    for i in range(k):
        if mm[i]:
            Xi = np.array(X[y==i])
            yi = alg.fit_predict(Xi)
            d = max(yi)
            yi[yi>0] = k+c+d
            yi[yi==0] = i
            y[y==i]=yi
            c = c+d
    return y


def rebuilt_labels(y):
    # inputs
    #   y: array with cluster labels (-1 for outliers) 
    # outputs
    #   y_new: refined array with cluster labels (-1 for outliers) 

    y_rem = np.unique(y)
    outs = np.where(y_rem == -1)
    a = 0
    if len(outs[0])>0:
        a=1
    y_new = np.copy(y)
    for i in range(0,y_rem.shape[0]-a):
        y_new[y==y_rem[i+a]]=i
    return y_new


def graph_ref(X,y,kinship):

    kinship[kinship==5]=0
    G = nx.from_numpy_matrix(kinship)

    kin = []
    for tup in list(G.edges):
        kin.append(5-kinship[tup[0],tup[1]])

    pos = nx.spring_layout(G)
    for edge in list(G.edges):
        if G[edge[0]][edge[1]]["weight"] == 4: 
            G.remove_edge(edge[0], edge[1])
        elif G[edge[0]][edge[1]]["weight"] == 3: 
            if multimodality(np.vstack((X[y==edge[0]],X[y==edge[1]]))):
                G.remove_edge(edge[0], edge[1])

    lsubG = list(nx.connected_components(G))

    if len(lsubG)==1:
        for edge in list(G.edges):
            if G[edge[0]][edge[1]]["weight"] >= 2: 
                G.remove_edge(edge[0], edge[1])

    lsubG = list(nx.connected_components(G))
    ynew = np.zeros(len(y), dtype=int)
    ynew[y==-1]=-1

    nc = 0
    for subG in lsubG:
        for lab in subG:
            ynew[y==lab] = nc
        nc = nc+1

    return ynew

def reassign_outliers(X,y,out_sens,centroids,extR):
    if out_sens==0:
        mem_th = np.ones(len(extR))*np.inf
    else:
        mem_th = np.divide(extR,out_sens)
    Xi = X[y==-1]
    dm = distance.cdist(Xi,centroids)
    yout = np.argmin(dm, axis=1)
    dm_min = np.min(dm, axis=1) 
    ths_ext = mem_th[yout]
    yout[dm_min>ths_ext]=-1
    ynew = y
    ynew[y==-1]=yout
    return ynew

def refine(X,y,cc,gv,rc,rep = REP, min_rdens = MIN_REL_DENSITY, min_mass = MIN_CARDINALITY_R, out_sens = OUTLIER_SENS):
    # inputs
    #   X: dataset (nxm), matrix of n vectors with m dimensions
    #   y: array with cluster labels (-1 for outliers)
    #   cc: cluster context (ClusterContext)
    #   gv: goi validity indices (GValidty) 
    #   rc: cluster refinement context (RefinementContext)
    #   rep: number of repetitions
    # outputs
    #   y: (refined?) array with cluster labels (-1 for outliers) 

    n, m = X.shape
    for j in range(0,rep+1):
        if sum(rc.mm):
            y = dig_multimodal(X,y,rc.mm)
            cc = cluster_context(X,y)
            gv = gval(cc)
            rc = refinement_context(X,y,cc,gv)

        ch_flag = False
        for i in range(0,cc.k):
            if cc.outliers < 0.5 and (rc.kdens[i] <= min_rdens or cc.mass[i] < sum(cc.mass) * min_mass):
                y[y==i] = -1 # or reassign them
                ch_flag = True

        if ch_flag:
            y = rebuilt_labels(y)
            cc = cluster_context(X,y)
            gv = gval(cc)
            rc = refinement_context(X,y,cc,gv)

        y = graph_ref(X,y,rc.kinship)

        cc = cluster_context(X,y)
        gv = gval(cc)
        rc = refinement_context(X,y,cc,gv)

        if sum(y==-1):
            y = reassign_outliers(X,y,out_sens,cc.centroids,gv.strR)
            cc = cluster_context(X,y)

        if j<rep:
            gv = gval(cc)
            rc = refinement_context(X,y,cc,gv)

    return y,cc


def cluster_kinship(k,De,erad,srad):
=======
def dataset_report(X,y,k,dataset,clusters,Odens,kdens,mm,kinship):
    # inputs
    #   X: dataset (nxm), matrix of n vectors with m dimensions
    #   y: array with cluster labels (-1 for outliers) 
    #   k: number of clusters (scalar)
    #   dataset: (dict) contains validity indices of the whole dataset
    #   clusters: (dict) contains validity indices of individual clusters
    #   kdens: cluster relative densities (kx1-array)
    #   Odens: global/overall density (scalar)
    #   mm: multimodality flags for each cluster (kx1-array)
    #   kinship: cluster kinship indices (k x k matrix): 4-unrelated, 3-friends, 2-relatives, 1-parent and child

    print("\nDataset info extracted from predictions and validation")
    print("======================================================\n")
    print("Dataset size:", X.shape)
    print("Number of outliers:", np.sum(X[y==-1]))
    Gstr = dataset['Gstr']
    Grex = dataset['Grex']
    Gmin = dataset['Gmin']
    print("G(lobal)OI strict:", Gstr)
    print("G(lobal)OI relax:", Grex)
    print("G(lobal)OI min:", Gmin)
    print("Overall density:", Odens)
    print("Number of clusters:", k)
    for i in range(0,k):
        print("\nCluster",i, "...")
        print("\toi strict:", clusters[i]['oi_st'])
        print("\toi relax:", clusters[i]['oi_rx'])
        print("\toi min:", clusters[i]['oi_mn'])
        print("\tcardinality:", clusters[i]['oi_mn'])
        print("\trelative density:", kdens[i])
        if mm[i] == 1:
            print("\tis multimodal!")

        for j in range(0,k):
            if i != j:
                if kinship[i,j] == 4:
                    print("\t... UNRELATED to cluster", j)
                if kinship[i,j] == 3:
                    print("\t... FRIEND of cluster", j)
                elif kinship[i,j] == 2:
                    print("\t... RELATIVE of cluster", j)
                elif kinship[i,j] == 1:
                    if clusters[i]['erad'] > clusters[j]['erad']:
                        print("\t... PARENT of cluster", j)
                    else:
                        print("\t... CHILD of cluster", j)

    print("\nSummary:")
    if Gstr>1:
        print("Good solution! The dataset seems to be clearly representable in a cluster-like structure and the algorithm satisfactorily solved the task.")
    elif Gstr>=0:
        if Grex>1:
            print("Space with noise or with clusters underlaid by distributions with density differences or slow density drops in the external layers.")
        elif Grex>=0:
            print("The solution is acceptable but clusters are vague, not highly consistent or too close to each other.")
    else:
        if Grex<0:
            print("The solution is not satisfactory. Either the input space is too complex, noisy or chaotic; or the algorithm is not performing properly.")
        elif Grex<1:
            print("There are distinctive points of the space with a higher density of objects.")
        else:
            print("Common in noisy spaces with well-defined density cores.")
    if Gmin>=0:
        print("There is no cluster overlap.")
    else:
        print("At least two clusters overlap.")

def cluster_kinship(k,De,erad):
>>>>>>> c182d4069ead97e8078f49ef0493d0ff9a1ea816
    # inputs
    #   k: number of clusters
    #   De: cluster inter distance matrix (k x k matrix)
    #   erad: extended radii (k-size array)
<<<<<<< HEAD
    #   srad: strict radii (k-size array)
    # outputs
    #   kinship: cluster kinship indices (k x k matrix): 5-unrelated, 4-acquitances, 3-close-friends, 2-relatives, 1-parent and child, 0-itself.
    kinship = np.zeros(shape=(k,k))
    comb = combinations(np.arange(k), 2) 

    for (i,j) in list(comb): 
        if erad[i] + erad[j] <= De[i,j]: 
            kinship[i,j],kinship[j,i] = 5,5
        else:
            if (erad[i] < De[i,j] and erad[j] < De[i,j]): #friends
                if ((De[i,j] - srad[i] < erad[j]) or (De[i,j] - srad[j] < erad[i])): #close friends
                    kinship[i,j],kinship[j,i] = 3,3
                else:
                    kinship[i,j],kinship[j,i] = 4,4 # acquaintance
            elif ((erad[i] + De[i,j] < erad[j]) or (erad[j] + De[i,j] < erad[i])): #parent and child
                kinship[i,j],kinship[j,i] = 1,1
            else: #relatives                 
                kinship[i,j],kinship[j,i] = 2,2
=======
    # outputs
    #   kinship: cluster kinship indices (k x k matrix): 4-unrelated, 3-friends, 2-relatives, 1-parent and child
    kinship = np.zeros(shape=(k,k))
    for i in range(0,k):
        for j in range(i+1,k):
            interD = De[i,j]
            radA = np.asscalar(erad[i])
            radB = np.asscalar(erad[j])
            if radA + radB <= interD: #unrelated
                kinship[i,j],kinship[j,i] = 4,4
            else:
                if (radA < interD and radB < interD): #friends
                    kinship[i,j],kinship[j,i] = 3,3
                elif ((radA + interD < radB) or (radB + interD < radA)): #parent and child
                    kinship[i,j],kinship[j,i] = 1,1
                else: #relatives                 
                    kinship[i,j],kinship[j,i] = 2,2
>>>>>>> c182d4069ead97e8078f49ef0493d0ff9a1ea816
    return kinship

def get_centroids(X,y,k):
    # inputs
    #   X: dataset (nxm), matrix of n vectors with m dimensions
    #   y: array with cluster labels (-1 for outliers) 
    #   k: number of clusters
    # outputs
    #   centroids: matrix (kxm) with robust cluster centers
    m = X.shape[1]
    centroids = np.zeros(shape=(k,m))
    for i in range(0,k):
        Xi = np.array(X[y==i])
        cXi = np.nanmedian(Xi, axis=0)
        centroids[i] = cXi
    return centroids

def multimodality(Xi):
    # inputs
    #   Xi: cluster data (nxm), matrix of n vectors with m dimensions
    # outputs
    #   mm: multimodality flag (scalar: 0 or 1)
    n, m = Xi.shape
<<<<<<< HEAD
    mm, bwf = 0,10
    points = int(50*(np.log10(n)+1))
    for i in range(0,m):
        feat = Xi[:,i].reshape(-1,1)
        bw=(max(feat)-min(feat))/bwf
        if bw > 0:
            x, y = FFTKDE(bw='silverman').fit(feat).evaluate(points)
            #x, y = FFTKDE(bw=bwf).fit(feat).evaluate(points)
            peaks, _ = find_peaks(y, prominence=0.5)
=======
    mm = 0
    bwf = 8
    for i in range(0,m):
        feat = Xi[:,i].reshape(-1,1)
        bw=(np.max(feat)-np.min(feat))/bwf
        if bw > 0:
            kde = KernelDensity(kernel='gaussian', bandwidth=bw, leaf_size=100).fit(feat)
            xbasis = np.linspace(np.min(feat), np.max(feat), 5*n)[:, np.newaxis]
            Xkde = np.exp(kde.score_samples(xbasis)) 
            peaks, _ = find_peaks(Xkde)
>>>>>>> c182d4069ead97e8078f49ef0493d0ff9a1ea816
            if len(peaks) > 1:
                mm = 1
    return mm

def multimodal_clusters(X,y,k):
    # inputs
    #   X: dataset (nxm), matrix of n vectors with m dimensions
    #   y: array with cluster labels (-1 for outliers) 
    #   k: number of clusters
    # outputs
    #   mm: multimodality flags for each cluster (kx1-array)
    mm = np.zeros(shape=(k,1))
    for i in range(0,k):
        Xi = np.array(X[y==i])
        if Xi.shape[0] > 0:
            mm[i] = multimodality(Xi)
    return mm

def rdensity(X, y, k):
    # inputs
    #   X: dataset (nxm), matrix of n vectors with m dimensions
    #   y: array with cluster labels (-1 for outliers) 
    #   k: number of clusters
    # outputs
    #   Odens: global/overall density (scalar)
    #   kdens: cluster relative densities (kx1-array)

    Ocentroid = np.nanmedian(X, axis=0)
    dXtoO = distance.cdist(X,[Ocentroid])
    Odens = 1/((np.nanmean(dXtoO) + 2*np.nanstd(dXtoO)) / X.shape[0])
    kdens = np.zeros(shape=(k,1))
    for i in range(0,k):
        Xi = np.array(X[y==i])
        cXi = np.nanmedian(Xi, axis=0)
        intradXi = distance.cdist(Xi,[cXi])
        medinXi = np.nanmedian(intradXi)
<<<<<<< HEAD
        if medinXi == 0:
            medinXi = 1
        icard = np.sum(y==i)
        kdens[i] = -1 + (icard/medinXi)/Odens
    return kdens, Odens

def refinement_context(X,y,cc,gv):
    # inputs
    #   X: dataset (nxm), matrix of n vectors with m dimensions
    #   y: array with cluster labels (-1 for outliers) 
    #   cc: cluster context (ClusterContext)
    #   gv: goi validity indices (GValidty)
    # outputs
    #   rc: cluster refinement context (RefinementContext)

    rc = RefinementContext(cc.k)
    rc.kdens, rc.Odens = rdensity(X,y,cc.k)
    rc.kinship = cluster_kinship(cc.k,cc.De,gv.extR,gv.strR)
    rc.mm = multimodal_clusters(X,y,cc.k)
    return rc

def other_validations(X,y,verbose = False):
    # inputs
    #   X: dataset (nxm), matrix of n vectors with m dimensions
    #   y: array with cluster labels (-1 for outliers) 
    #   verbose: (bool) "True" stands for verbose mode
    # outputs
    #   S: Silhouette index of the whole dataset (float) 
    #   CH: Calinski Harabasz index of the whole dataset (float) 
    #   DB: Davies Bouldin of the whole dataset (float) 

    from sklearn import metrics
    X=X[y!=-1,:]
    y=y[y!=-1]
    S,CH,DB = np.nan, np.nan, np.nan
    if len(y):
        k = max(y)
        if k>0: 
            S = metrics.silhouette_score(X, y, metric='euclidean')
            CH = metrics.calinski_harabasz_score(X, y)
            DB = metrics.davies_bouldin_score(X, y)
    if verbose:
        print("- Validity index > Silhouette:", S)
        print("- Validity index > Calinski Harabasz:", CH)
        print("- Validity index > Davies Bouldin:", DB)
    return S,CH,DB

def gval(cc, verbose = False):
    # inputs
    #   cc: cluster context (ClusterContext)
    #   verbose: (bool) "True" stands for verbose mode
    # outputs
    #   gv: goi validity indices (GValidty)

    k = cc.k
    gv = GValidity(k)

    radm, radm2, = np.zeros(shape=(k,1)), np.zeros(shape=(k,1))
    oist, oirx = np.ones(shape=(k,k))*np.inf, np.ones(shape=(k,k))*np.inf

    gv.extR = cc.mnDa + 2*cc.sdDa 
    gv.strR = cc.mdDa
    gv.volR = np.divide(gv.extR, gv.strR, out=np.ones_like(gv.extR), where=gv.strR!=0)
    radm = np.multiply(gv.extR, cc.mass)
    radm2 = np.multiply(gv.strR, cc.mass)

    per = permutations(np.arange(cc.k), 2) 

    for (i,j) in list(per): 
        oist[i][j] =  cc.De[i][j] - gv.extR[i] - (cc.mnDa[j] +2*cc.sdDa[j])
        oirx[i][j] = cc.De[i][j] - cc.mdDa[i] - cc.mdDa[j]

    gv.oi_st = np.amin(oist, axis=0)
    gv.oi_rx = np.amin(oirx, axis=0)    
    gv.oi_mn = np.divide(gv.oi_st, gv.extR, out=gv.oi_st, where=gv.extR!=0)

    gv.Gstr = np.sum(np.multiply(gv.oi_st, cc.mass)) / np.sum(radm)
    gv.Grex = np.sum(np.multiply(gv.oi_rx, cc.mass)) / np.sum(radm2)
    if len(gv.oi_mn):
        gv.Gmin = np.nanmin(gv.oi_mn)

    if verbose:
        print("- Validity index > GOI > Grex:", gv.Grex)
        print("- Validity index > GOI > Gstr:", gv.Gstr)
        print("- Validity index > GOI > Gmin:", gv.Gmin)
    return gv


def cluster_context(X,y):
    # inputs
    #   X: dataset (nxm), matrix of n vectors with m dimensions
    #   y: array with cluster labels (-1 for outliers) 
    # outputs
    #   ClusterContext: cluster context

    k = max(y)+1
    cc = ClusterContext(k,len(X[0,:]))
    cXi = np.zeros(shape=(k,X.shape[1]))
    for i in range(0,k):
        Xi = np.array(X[y==i])
        cc.mass[i] = Xi.shape[0]
        cX = np.nanmedian(Xi, axis=0)
        cXi[i] = cX
        dm = distance.cdist(Xi,[cX])
        cc.mnDa[i] = np.nanmean(dm)
        cc.mdDa[i] = np.nanmedian(dm)
        cc.sdDa[i] = np.nanstd(dm)
    cc.De = distance_matrix(cXi,cXi)
    cc.centroids = cXi
    cc.outliers = sum(y==-1)/sum(cc.mass)
    return cc

=======
        icard = np.sum(y==i)
        kdens[i] = -1 + (icard/medinXi)/Odens
    return Odens, kdens

def extract_cluster_context(X,y):
    # inputs
    #   X: dataset (nxm), matrix of n vectors with m dimensions
    #   y: array with cluster labels (-1 for outliers) 
    # outputs
    #   k: number of clusters (scalar)
    #   mass: cluster mass or cardinality (k-size array)
    #   mnDa: cluster mean intra distance (k-size array)
    #   mdDa: cluster median intra distance (k-size array)
    #   sdDa: cluster std-dev intra distance (k-size array)
    #   De: cluster inter distance matrix (k x k matrix)

    maxID = max(y)
    k = maxID +1
    mdDa, mnDa, sdDa, mass = np.zeros(shape=(k,1)), np.zeros(shape=(k,1)), np.zeros(shape=(k,1)), np.zeros(shape=(k,1))
    cXi = np.zeros(shape=(k,X.shape[1]))
    for i in range(0,k):
        Xi = np.array(X[y==i])
        mass[i] = Xi.shape[0]
        cX = np.nanmedian(Xi, axis=0)
        cXi[i] = cX
        dm = distance.cdist(Xi,[cX])
        mnDa[i] = np.nanmean(dm)
        mdDa[i] = np.nanmedian(dm)
        sdDa[i] = np.nanstd(dm)
    De = distance_matrix(cXi,cXi)
    return k,De,mdDa,mnDa,sdDa,mass

def gval(k,De,mdDa,mnDa,sdDa,mass):
    # inputs
    #   k: number of clusters (scalar)
    #   mass: cluster mass or cardinality (k-size array)
    #   mnDa: cluster mean intra distance (k-size array)
    #   mdDa: cluster median intra distance (k-size array)
    #   sdDa: cluster std-dev intra distance (k-size array)
    #   De: cluster inter distance matrix (k x k matrix)
    # outputs
    #   dataset: 
    #       Gstr, Grex, Gmin: Goi global indices (scalars)
    #   clusters: 
    #       oi_st, oi_rx, oi_mn: Goi cluster indices (scalars)
    #       volratio: times that the extended radious is in the core radious (scalar)
    #   erad: extended radii (k-size array)
    #   volr: times that extended radii are in their respective core radii (k-size array)
    #   oimn: oi_min of clusters (k-size array)

    dataset = {'Gstr':0, 'Grex':0, 'Gmin':0}
    rad, rad2, radm, radm2, oimn, erad, volr = np.zeros(shape=(k,1)), np.zeros(shape=(k,1)), np.zeros(shape=(k,1)), np.zeros(shape=(k,1)), np.zeros(shape=(k,1)), np.zeros(shape=(k,1)), np.zeros(shape=(k,1))
    oist, oirx = np.zeros(shape=(k-1,1)), np.zeros(shape=(k-1,1))
    clusters = {}

    for i in range(0,k):
        clusters[i] = {}
        clusters[i]['oi_st'] = 0
        clusters[i]['oi_rx'] = 0
        clusters[i]['oi_mn'] = 0
        rad[i] = mnDa[i] + 2*sdDa[i]
        rad2[i] = mdDa[i]
        clusters[i]['volratio'] = rad[i] / rad2[i]
        clusters[i]['erad'] = rad[i] 
        clusters[i]['srad'] = rad2[i]
        erad[i] = clusters[i]['erad']
        volr[i] = clusters[i]['volratio']
        radm[i] = rad[i] * mass[i];
        radm2[i] = rad2[i] * mass[i];

    if k>1:
        for i in range(0,k):
            l=0
            for j in range(0,k):
                if j != i:
                    oist[l] = De[i][j] - rad[i] - (mnDa[j] +2*sdDa[j])
                    oirx[l] = De[i][j] - mdDa[i] - mdDa[j]
                    l=l+1;
        
            clusters[i]['oi_st'] = np.nanmin(oist)
            clusters[i]['oi_rx'] = np.nanmin(oirx)
            if mnDa[i]>0:
                clusters[i]['oi_mn'] = np.nanmin(oist) / (mnDa[i] + 2*sdDa[i])
            else:
                clusters[i]['oi_mn'] = np.nanmin(oist)
         
            dataset['Gstr'] = dataset['Gstr'] + clusters[i]['oi_st']*mass[i]
            dataset['Grex'] = dataset['Grex'] + clusters[i]['oi_rx']*mass[i]
            oimn[i] = clusters[i]['oi_mn']

        dataset['Gmin'] = np.nanmin(oimn)
        dataset['Gstr'] = dataset['Gstr'] / np.sum(radm)
        dataset['Grex'] = dataset['Grex'] / np.sum(radm2)
    else:
        dataset['Gmin'] = np.nan
        dataset['Gstr'] = np.nan
        dataset['Grex'] = np.nan

    return dataset,clusters,erad,volr,oimn

def clureal_complete(X, y, refinement=False, repetitions=REP, report=False, SK=False):
    # inputs
    #   X: dataset (nxm), matrix of n vectors with m dimensions
    #   y: array with cluster labels (-1 for outliers) 
    #   refinement: (bool) if the CluReAL algorithm must be used
    #   repetitions: (scalar) the number of times to iterate CluReAL
    #   report: (bool) "True" stands for verbose mode
    #   SK: (bool) "True" draws SK symbols
    # outputs
    #   y: (refined?) array with cluster labels (-1 for outliers) 
    #   dataset: (dict) contains validity indices of the whole dataset
    #   clusters: (dict) contains validity indices of individual clusters

    np.seterr(divide='ignore')
    np.seterr(invalid='ignore')
    k,De,mdDa,mnDa,sdDa,mass = extract_cluster_context(X,y)
    centroids = get_centroids(X,y,k)
    Odens,kdens = rdensity (X,y,k)
    mm = multimodal_clusters (X,y,k)
    dataset, clusters, erad, volr, oimin = gval(k,De,mdDa,mnDa,sdDa,mass)
    kinship = cluster_kinship(k,De,erad)

    if refinement:
        for i in range(repetitions):
            y = cluster_refinement(X,y,k,mm,kdens,Odens,kinship,mass,De,volr,oimin)
            k,De,mdDa,mnDa,sdDa,mass = extract_cluster_context(X,y)
            Odens,kdens = rdensity (X,y,k)
            mm = multimodal_clusters (X,y,k)
            dataset, clusters, erad, volr, oimin = gval(k,De,mdDa,mnDa,sdDa,mass)
            kinship = cluster_kinship(k,De,erad)

    if report:
        dataset_report(X,y,k,dataset,clusters,Odens,kdens,mm,kinship)

    dataset['Silhouette'], dataset['Calinski Harabasz'], dataset['Davies Bouldin'] = other_validations(X,y,verbose=report)
    outliers = np.sum(y==-1)/np.sum(mass)

    if SK:
        draw_symbol(k,dataset,clusters,mm,kinship,kdens,volr,outliers)

    return y, dataset, clusters

if __name__== '__main__':
    from sklearn.datasets.samples_generator import make_blobs
    X, y_real = make_blobs(n_samples=1500, centers=7, n_features=2, random_state=0, cluster_std=0.6)

    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler(copy=True, feature_range=(0, 1))
    scaler.fit(X)
    X = scaler.transform(X)

    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=10, random_state=0).fit(X)
    y = kmeans.predict(X)

    plt.figure(figsize=(9 * 2 + 3, 12.5))
    plt.subplots_adjust(left=.02, right=.98, bottom=.001, top=.96, wspace=.05, hspace=.01)

    from itertools import cycle,islice
    colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a', '#f781bf', '#a65628', '#984ea3', '#999999', '#e41a1c', '#dede00', '#b56537']),int(max(y) + 1))))
    # add black color for outliers (if any)
    colors = np.append(colors, ["#000000"])

    plt.subplot(2, 3, 1)
    plt.scatter(X[:, 0], X[:, 1], s=1)
    plt.title("Original dataset")

    plt.subplot(2, 3, 2)
    plt.scatter(X[:, 0], X[:, 1], s=1, color=colors[y])
    plt.title("Dataset after k-means clustering (k=10)")

    plt.subplot(2, 3, 5)
    clureal_complete(X,y,SK=True, report=True)

    plt.subplot(2, 3, 6)
    y, _, _ = clureal_complete(X,y,refinement=True,SK=True, report=True)

    plt.subplot(2, 3, 3)
    plt.scatter(X[:, 0], X[:, 1], s=1, color=colors[y])
    plt.title("Dataset after k-means + CluReAL refinement")

    plt.show()    
>>>>>>> c182d4069ead97e8078f49ef0493d0ff9a1ea816

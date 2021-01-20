"""
======================
CluReAL algorithm v2.0
FIV, Nov 2020
======================
"""
#!/usr/bin/env python3

import numpy as np
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

def draw_symbol(cc, gv, rc):
    # inputs
    #   cc: cluster context (ClusterContext)
    #   gv: goi validity indices (GValidty) 
    #   rc: cluster refinement context (RefinementContext)

    k, outliers = cc.k, cc.outliers
    Gstr, Grex, Gmin, volr = gv.Gstr, gv.Grex, gv.Gmin, gv.volR
    mm,kinship,kdens = rc.mm, rc.kinship, rc.kdens

    child = np.where(kinship == 1)
    densdiff = np.absolute(np.nanmax(kdens)-np.nanmin(kdens))/np.absolute(np.minimum( np.nanmax(kdens),np.nanmin(kdens) ))  
    volr = np.nanmean(volr)

    x_ec, x_ec2, x_cc, x_cc2, x_ccup, x_ech = 0, 0, 0, 0, 0, 0 
    y_ec, y_cc = 0, 0
    v_ec, v_ec2, v_cc, v_cc2, v_ccup, v_ech, v_r1, v_l1, f_ec2, v_eov = False, False, False, False, False, False, False, False, False, False
    v_ol, v_om, v_oh = False, False, False
    c_ec2 = 'k' 

    if (sum(mm)>0):
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
    # inputs
    #   k: number of clusters
    #   De: cluster inter distance matrix (k x k matrix)
    #   erad: extended radii (k-size array)
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
    mm, bwf = 0,10
    points = int(50*(np.log10(n)+1))
    for i in range(0,m):
        feat = Xi[:,i].reshape(-1,1)
        bw=(max(feat)-min(feat))/bwf
        if bw > 0:
            x, y = FFTKDE(bw='silverman').fit(feat).evaluate(points)
            #x, y = FFTKDE(bw=bwf).fit(feat).evaluate(points)
            peaks, _ = find_peaks(y, prominence=0.5)
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


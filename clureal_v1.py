"""
======================
CluReAL algorithm v1.0
???, Jun 2020
======================

"""
#!/usr/bin/env python3

import numpy as np
from scipy.spatial import distance_matrix
from scipy.spatial import distance
from sklearn.neighbors import KernelDensity
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
    child = np.where(kinship == 1)
    densdiff = np.absolute(np.nanmax(kdens)-np.nanmin(kdens))/np.absolute(np.minimum( np.nanmax(kdens),np.nanmin(kdens) ))  
    volr = np.nanmean(volr)

    x_ec, x_ec2, x_cc, x_cc2, x_ccup, x_ech = 0, 0, 0, 0, 0, 0 
    y_ec, y_cc = 0, 0
    v_ec, v_ec2, v_cc, v_cc2, v_ccup, v_ech, v_r1, v_l1, f_ec2, v_eov = False, False, False, False, False, False, False, False, False, False
    v_ol, v_om, v_oh = False, False, False
    c_ec2 = 'k' 

    if (np.sum(mm)>0):
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
    # inputs
    #   k: number of clusters
    #   De: cluster inter distance matrix (k x k matrix)
    #   erad: extended radii (k-size array)
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

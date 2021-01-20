
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

import clureal as cr

np.random.seed(100)

def get_cmap(n, name='hsv'):
    return plt.cm.get_cmap(name, n)

# Creating a dataset to cluster
from sklearn.datasets import make_blobs
X, y_real = make_blobs(n_samples=1500, centers=7, n_features=2, random_state=0, cluster_std=0.6)

# Normal clustering with KMeans
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=10, random_state=0).fit(X)
y = kmeans.predict(X)

# Plotting environment
plt.figure(figsize=(9 * 2 + 3, 12.5))
plt.subplots_adjust(left=.02, right=.98, bottom=.001, top=.96, wspace=.05, hspace=.01)

# Plotting original data
plt.subplot(2, 3, 1)
plt.scatter(X[:, 0], X[:, 1], s=2)
plt.title("Original dataset")

# Plotting clustered data with normal k-means
plt.subplot(2, 3, 2)
cmap = get_cmap(max(y+2),'tab20b')
plt.scatter(X[:, 0], X[:, 1], s=2, color=cmap(y+1))
plt.title("Dataset after k-means clustering (k=10)")

# Extracting clustering metadata for the SK ideogram
cc = cr.cluster_context(X,y)
gv = cr.gval(cc)
rc = cr.refinement_context(X,y,cc,gv)

# Plotting the SK ideogram of the normal clustering
plt.subplot(2, 3, 5)
cr.draw_symbol(cc, gv, rc)

# Refining with CluReAL and updating metadata
y,cc = cr.refine(X,y,cc,gv,rc)
gv = cr.gval(cc)
rc = cr.refinement_context(X,y,cc,gv)

# Plotting the refined clustered data
plt.subplot(2, 3, 3)
cmap = get_cmap(max(y+2),'tab20b')
plt.scatter(X[:, 0], X[:, 1], s=2, color=cmap(y+1))
plt.scatter(X[y==-1, 0], X[y==-1, 1], s=2, c='k')
plt.title("Dataset after k-means + CluReAL refinement")

# Plotting the SK ideogram of the normal clustering
plt.subplot(2, 3, 6)
cr.draw_symbol(cc, gv, rc)



plt.show()    




cc = cr.cluster_context(X,y)
gv = cr.gval(cc)
cr.draw_symbol(cc, gv, rc)




















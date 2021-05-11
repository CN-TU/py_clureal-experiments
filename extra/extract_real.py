"""
==============================================
Extracting tSNE projections from real datasets
 
FIV, May 2021
==============================================
"""

import pandas as pd
import sys
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn import datasets
import matplotlib.pyplot as plt

d1 = datasets.load_breast_cancer()
d2 = datasets.load_iris()
d3 = datasets.load_digits()
d4 = datasets.load_wine()

datas = [d1,d2,d3,d4]
names = ['real_1','real_2','real_3','real_4']

for i,data in enumerate(datas):
    name = names[i]
    y = data['target']
    X = data['data']

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    from sklearn import manifold
    mds = manifold.TSNE(n_components=2, random_state=5, init='pca', n_jobs=-1, early_exaggeration=100)
    X = mds.fit_transform(X)

    df = pd.DataFrame(data=X)
    df['label']=y

    plt.scatter(X[:,0],X[:,1],c=y)
    plt.show()

    df.to_csv(name, sep = ',', index = False, header = False)




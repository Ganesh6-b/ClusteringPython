# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 17:33:08 2019

@author: Ganesh
"""

import pandas as pd

data = pd.read_csv("F:\\R\\files\\crime_data.csv")
data["Unnamed: 0"].value_counts()
df = data.iloc[:,1:]
#checking na's
df.isna().sum()

#scalling
from sklearn import preprocessing

df.iloc[:,1:] = preprocessing.normalize(df.iloc[:,1:])

#clustering

from scipy.cluster.hierarchy import linkage

import scipy.cluster.hierarchy as sch

type(df)

a = linkage(df, method = "complete", metric = "euclidean")
a
import matplotlib.pyplot as plt
sch.dendrogram(a);plt.figure(figsize=(15,5))

#agglomerative clustering

from sklearn.cluster import AgglomerativeClustering

h_complete = AgglomerativeClustering(n_clusters = 3, linkage = "complete", affinity = "euclidean").fit(df)

h_complete.labels_

type(h_complete.labels_)

cluster_labels = pd.Series(h_complete.labels_)

data['Clust'] = cluster_labels

data = data.iloc[:,[5,0,1,2,3,4]]

#getting aggregate values

data.groupby(data.Clust).mean()

#for kmeans clustering 
##
from sklearn.cluster import KMeans

from scipy.spatial.distance import cdist

#screwplot
k = list(range(2,15))
k
TWSS = []
for i in k:
    Kmeans = KMeans(n_clusters = i).fit(df)
    wss = []
    for j in range(i):
        wss.append(sum(cdist(df.iloc[Kmeans.labels_ == j,:], Kmeans.cluster_centers_[j].reshape(1,df.shape[1]),"euclidean")))
        TWSS.append(sum(wss))
plt.plot(k,TWSS, "-ro")

m1 = KMeans(n_clusters = 3).fit(df)

labels = m1.labels_

labels

data.groupby(labels).mean()

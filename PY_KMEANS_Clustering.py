#!/usr/bin/env python
# coding: utf-8

# In[45]:


##implementation of a simple Kmeans Clustering model very basic using IRIS dataset.

#http://seaborn.pydata.org/generated/seaborn.scatterplot.html?highlight=s
#https://stackabuse.com/k-means-clustering-with-scikit-learn/
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from pandas import DataFrame
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans


# In[47]:


iris = pd.read_csv('IRIS.csv')
df = DataFrame(iris,columns=['sepal_length','petal_length'])
kmeans = KMeans(n_clusters=3).fit(df)
centroids = kmeans.cluster_centers_
print(centroids)


# In[50]:


plt.scatter(df['sepal_length'], df['petal_length'], c= kmeans.labels_.astype(float), s=50, alpha=0.5)
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=100)


# In[ ]:





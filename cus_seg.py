# -*- coding: utf-8 -*-
"""
Created on Wed May 20 14:27:08 2020

@author: Sayan Mondal
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib qt
import seaborn as sns

data=pd.read_csv("C:/Users/Sayan Mondal/Desktop/cusomer segmentation/Mall_Customers.csv")

data.columns
data.shape
data.isnull().sum()
data.describe().T

### removing the customerid column...##
data.drop(["CustomerID"],axis=1,inplace=True)
data
cormat=data.corr()

### plotting the heat map...##

plt.rcParams['figure.figsize'] = (10,10)
hm=sns.heatmap(cormat,linewidths = 0.5, annot=True, center=0,cmap='twilight')
hm.set_title(label='Heatmap of dataset', fontsize=16)
hm

## Bar plot...##
# plot between 2 attributes ...###
plt.bar(data['Age'], data['Annual Income (k$)']) 
plt.xlabel("Age") 
plt.ylabel("Annual Income (k$)") 
plt.show()

plt.bar(data['Annual Income (k$)'], data['Spending Score (1-100)']) 
plt.xlabel("Annual Income (k$)") 
plt.ylabel("Spending Score (1-100)") 
plt.show()

## male and female ratio..###
ratio = data['Gender'].value_counts()
print(ratio)

labels=['Male','Female']
size=[88,112]
colors=['green','yellow']
explode=(0.2,0)
plt.pie(size,explode=explode,labels=labels, colors=colors,shadow=True)
plt.title('Male & Female Ratio')
plt.show()

##  Gender vs.Spending score..##
plt.rcParams['figure.figsize'] = (15, 8)
sns.boxenplot(data['Gender'], data['Spending Score (1-100)'], palette ="BuGn_r")
plt.title('Gender vs Spending Score', fontsize = 18)
plt.show()

## Gender's age vs spending score....##
plt.rcParams['figure.figsize'] = (15, 8)
sns.stripplot(data['Gender'], data['Age'], palette = 'BuGn_r', size = 10)
plt.title('Gender vs Spending Score', fontsize = 18)
plt.show()

##...Distribution of spending ..##
plt.rcParams['figure.figsize'] = (15, 8)
sns.countplot(data['Spending Score (1-100)'], palette = 'rainbow')
plt.title('Distribution of Spending Score', fontsize = 14)
plt.show()


######......... Average annual income of male & female...######################
print("Mean of Annual Income (k$) of Female:",data['Annual Income (k$)'].loc[data['Gender'] == 'Female'].mean())
print("Mean of Annual Income (k$) of Male:",data['Annual Income (k$)'].loc[data['Gender'] == 'Male'].mean())

### kde plot for male vs female income...#
p1=sns.kdeplot(data['Annual Income (k$)'].loc[data['Gender'] == 'Male'],label='Income Male', shade=True, color="g")
p1=sns.kdeplot(data['Annual Income (k$)'].loc[data['Gender'] == 'Female'],label='Income Female', shade=True, color="b")
plt.xlabel('Annual Income (k$)')
plt.show()

#############.... Average spending of male & female...###############
print("Mean of Spending Score (1-100) of Female:",data['Spending Score (1-100)'].loc[data['Gender'] == 'Female'].mean())
print("Mean of Spending Score (1-100) of Male:",data['Spending Score (1-100)'].loc[data['Gender'] == 'Male'].mean())

p1=sns.kdeplot(data['Spending Score (1-100)'].loc[data['Gender'] == 'Male'],label='Density Male',bw=2, shade=True, color="r")
p1=sns.kdeplot(data['Spending Score (1-100)'].loc[data['Gender'] == 'Female'],label='Density Female',bw=2, shade=True, color="b")
plt.xlabel('Genderwise Spending Score')
plt.show()

## Lets preprocess the data....####
## Converting Gender to lebel...###
from sklearn import preprocessing
le=preprocessing.LabelEncoder() 

data['Gender']=pd.get_dummies(data['Gender'])


from sklearn.preprocessing import MinMaxScaler
MMS=MinMaxScaler()

data[['Age','Annual Income (k$)','Spending Score (1-100)']]=MMS.fit_transform(data[['Age','Annual Income (k$)','Spending Score (1-100)']])


## Elbow plot....###
from sklearn.cluster import KMeans
X=data.loc[:,'Gender': 'Spending Score (1-100)']
k_range=range(1,15)
wcss=[]
for k in k_range:
    km=KMeans(n_clusters=k)
    km.fit(X)
    wcss.append(km.inertia_)

wcss

plt.xlabel('K')
plt.ylabel('WCSS')
plt.plot(k_range,wcss)
  

##  Now from the elbow plot,took number of cluster is 4...###
km=KMeans(n_clusters=4)
km
y_pred=km.fit_predict(data)
y_pred

data['cluster']=y_pred


final_clustered= data.to_csv("C:/Users/Sayan Mondal/Desktop/cusomer segmentation/clustered_file.csv")
# countplot to check the number of clusters and number of customers in each cluster
sns.countplot(y_pred)

plt.scatter(data[y_pred == 0, 0], data[y_pred == 0,1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(data[y_pred == 1, 0], data[y_pred== 1,1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(data[y_pred == 2, 0], data[y_pred == 2,1], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(data[y_pred == 3, 0], data[y_pred == 3,1], s = 100, c = 'cyan', label = 'Cluster 4')
plt.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1], s = 100, c = 'yellow', label = 'Centroids')
plt.title('Customers clusters')
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.legend()
plt.show()




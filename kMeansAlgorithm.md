**Linear Algebra and K-means clustering algorithm**

Zachary Bohn, Reuella Jacob, Matthew Niculae and Roberto Rodriguez

EGR 328, California Baptist University

April 26, 2022

**Abstract**

For this paper the group decided to explore linear algebra in the context of k-means clustering algorithms. The students explored clustering and the details of K-means clustering, various ways to initialize. The students also learned how to improve accuracy of K-means clustering by considering background information of features for any given data set. However, for the sake of simplicity this was only a theoretical approach. They then proceeded to implement K-means clustering on a random dataset. To improve the accuracy of K-means the students wrote a script to evaluate which value of K provided the most optimal results. Starting with this new value of K proved to be helpful in improving accuracy and reducing computation time of the algorithm.

# **Background**

We plan on exploring linear algebra in the context of clustering when considering data mining and web searches. According to Wagstaff, clustering algorithms are usually implemented when the dataset has no features associated with it. The data is classified on some &quot;notion of similarity&quot; and are considered unsupervised. Clustering is often used to measure if the given data set is useful. K-means is a type of partitional clustering algorithm. It is based on the idea that if we initially start with a sub-optimal clustering by dividing the data into K groups, but update the cluster centers to new means we calculate using the updated members. After a certain amount of iterations we will be able to accurately cluster the data. K-means is relatively simple to implement and scalable to large data sets. It does however cluster outliers and is quite dependent on initial values. This can be problematic if the dataset is not properly sanitized or there are a lot of outliers that can skew those clusters.

## **Project overview and Literature Review**

In this paper we will be looking at a random data set and implementing K-means clustering on it. There are various ways we can initialize the k clusters, some of them include &quot;Forgy initialization (randomly choosing K data points in the dataset) and Random partitions (dividing the data points randomly into K subsets), to more sophisticated methods, such as density-based initialization, Intelligent initialization, Furthest First initialization (FF for short, it works by picking the first center point randomly, then adding more center points which are furthest from existing ones), and subset furthest-first (SFF)initialization&quot; (Xin Jin &amp; Jiawei Han, 1970).

For the sake of simplicity we are ignoring any associated features that a real life data set might contain. However it is possible to modify K-means clustering to incorporate additional information about the features of the data set. Based on the research done by Wagstaff et al. doing so significantly increases the accuracy of the algorithm. Introducing background information does change the behavior of the algorithm and how it responds to the data. Wagstaff noted that the algorithm tends to become order-sensitive when accepting data. Ignorance in the early stages can lead to instances where the data clusters are not accurate but the algorithm is unable to rorganize the clusters.

### **Practical Implementation**

For this basic implementation, a random dataset was generated using a sklearn dataset. The randomized dataset generated was divided up into columns describing two distinct features.

x, y = make\_blobs(n\_samples=400, centers=3, random\_state=20, cluster\_std=1.1)

data = pd.DataFrame(data=x, columns=[&#39;feature1&#39;,&#39;feature2&#39;])

data.head()

data.describe()

![](RackMultipart20220711-1-rboya3_html_4ba06b147154a84.png)

In order to visualize this dataset, a scatterplot that graphs data based on (feature1, feature2) pairs was utilized.

fig = plt.figure(figsize=(15,8))

plt.scatter(x=data.feature1, y=data.feature2,

alpha=0.7, s=150)

ax = fig.get\_axes()[0]

ax.set\_axis\_off()

plt.show()

The result of this code segment was as follows:

![](RackMultipart20220711-1-rboya3_html_538619a3372abd3a.png)

The amount of centroids is randomly initialized in the following functions and randomly distributed throughout the scatterPlot

centroids = []

defcentroid\_init(data, ncentroid, figure=True):

centroids.clear()

random\_index = random.sample(range(0, data.shape[0]), ncentroid)

for i, index inenumerate(random\_index):

centroids.append(data.iloc[index].values.tolist())

if figure == True:

fig = plt.figure(figsize=(15,8))

sns.scatterplot(x=data.feature1, y=data.feature2,

alpha=0.8, s=200, color=&#39;grey&#39;)

for i inrange(len(centroids)):

plt.scatter(x=centroids[i][0], y=centroids[i][1], marker=&#39;\*&#39;, s=500, color=colors[i], linewidth=2, edgecolor=&#39;k&#39;)

ax = fig.get\_axes()[0]

ax.set\_axis\_off()

plt.show()

centroid\_init(data,3)

Centroid locations are denoted by stars that appear in different colors based on a previously initialized array

![](RackMultipart20220711-1-rboya3_html_bb39b3a0ee175a06.png)

In the following code, we measure the euclidean distance between each centroid and data point to assign the data point to a unique cluster classification.

cluster = []

euclidean\_distance = []

defeuclidean\_distance\_func(figure=True):

cluster.clear()

euclidean\_distance.clear()

for i inrange(data.shape[0]):

x = []

for centroid in centroids:

x.append(np.sqrt(np.dot(data.iloc[i].values-centroid, data.iloc[i].values-centroid)))

euclidean\_distance.append(x)

group = euclidean\_distance[i].index(min(euclidean\_distance[i]))

cluster.append(group)

if figure == True:

fig = plt.figure(figsize=(15,8))

sns.scatterplot(x=data.feature1, y=data.feature2,

alpha=0.6, s=200, hue=cluster, palette=palette)

for i, centroid inenumerate(centroids):

plt.scatter(x=centroid[0], y=centroid[1], marker=&#39;\*&#39;, s=500, color=colors[i], linewidth=2, edgecolor=&#39;k&#39;)

ax = fig.get\_axes()[0]

ax.set\_axis\_off()

plt.legend([])

plt.show()

euclidean\_distance\_func()

The cluster classification of these data points results in the following scatterPlot.

![](RackMultipart20220711-1-rboya3_html_f75d06f7a5e83c6.png)

Following classification of the data points, it is necessary that we find the mean of each cluster and assign the centroid to that mean.

new\_centroids = []

defmove\_centroids(figure=True):

new\_centroids.clear()

for i in np.unique(cluster):

df = data[np.array(cluster) == i]

centroid = [df.feature1.mean(), df.feature2.mean()]

new\_centroids.append(centroid)

if figure == True:

fig = plt.figure(figsize=(15,8))

sns.scatterplot(x=data.feature1, y=data.feature2,

alpha=0.6, s=200, hue=cluster, palette=palette)

for i, centroid inenumerate(new\_centroids):

plt.scatter(x=centroid[0], y=centroid[1], marker=&#39;\*&#39;, s=500, color=colors[i], linewidth=2, edgecolor=&#39;k&#39;)

ax = fig.get\_axes()[0]

ax.set\_axis\_off()

plt.legend([])

plt.show()

move\_centroids()

Centroid relocation to the cluster means results in the following distribution:

![](RackMultipart20220711-1-rboya3_html_bc68193b69c79713.png)

The centroids are now very far from convergence and as a result we need to repeat the following steps: 1.) Measure the Euclidean Distance between each datapoint and centroid 2.) Calculate the mean of each cluster and update the centroid with the mean of each cluster. This will be repeated until convergence or until the maximum number of iterations is reached.

for \_ inrange(10):

if new\_centroids == centroids:

break

else:

centroids = new\_centroids

new\_centroids = []

euclidean\_distance\_func()

move\_centroids()

The code returns the following succession of scatter plots:

1.) ![](RackMultipart20220711-1-rboya3_html_a05967f3a1c795a7.png)4.) ![](RackMultipart20220711-1-rboya3_html_f5aed184324d34db.png)

2.) ![](RackMultipart20220711-1-rboya3_html_44aa3eca1bf9afc.png)5.) ![](RackMultipart20220711-1-rboya3_html_f5aed184324d34db.png)

3.) ![](RackMultipart20220711-1-rboya3_html_ec73f30f71a84fe3.png)6.) ![](RackMultipart20220711-1-rboya3_html_f5aed184324d34db.png)

### **Optimization**

From the given code, we are able to run the unobserved k-means model to optimize the data points&#39; cluster classification and centroid position, however the algorithm itself does not give us a way to find the optimal amount of clusters to allocate. However, given an optimization implemented in python, we are able to find a relationship between WCSS score (within cluster sum of squares) and the number of clusters to use. WCSS is the sum of squared distance between each point and the centroid in a cluster. The following code iterates through the number of clusters and runs the WCSS score for each to determine the optimal solution.

wcss\_list = []

for n\_cluster inrange(1,10):

centroid\_init(data, n\_cluster, figure=False)

euclidean\_distance\_func(figure=False)

move\_centroids(figure=False)

for \_ inrange(10):

if new\_centroids == centroids:

break

else:

centroids = new\_centroids

new\_centroids = []

euclidean\_distance\_func(figure=False)

move\_centroids(figure=False)

wcss = 0

for i in np.unique(cluster):

icluster = 0

ddf = data[np.array(cluster) == i]

for index in ddf.index:

icluster += (euclidean\_distance[index][cluster[index]]\*\*2)

wcss += icluster

wcss\_list.append(wcss)

Plotting these relationships would yield the following curve:

![](RackMultipart20220711-1-rboya3_html_911db6a512fe5800.png)

This curve tells us that the amount of clusters that is optimal for this dataset is 3. We are able to run this method instead of just guessing and checking to attempt an optimal solution. Given this optimization, the k-means model would require much less computing for an optimal solution.

### **Conclusion**

In the course of this paper, we explored a basic implementation of the K-means model on. As a result of it being an unsupervised algorithm, the K-means model can be run on data with a lot less descriptive information which is optimal for market segmentation, document clustering, image segmentation, and image compression. Upon implementing this model, it was discovered that we could optimize the setup process by including a statistic to determine the optimal amount of clusters within the dataset. For this randomized dataset, the clusters seemed apparent but that&#39;s because it was initialized that way. When running a model on other datasets, the optimal amount of clusters will not be as clear. Finding the WCSS score prior to centroid classification would be the optimal route to follow. The computing power that it takes to calculate the WCSS is negligible compared to that of other cluster estimation strategies. This optimization will help with one of the K-means model major disadvantages: estimating the optimal number of clusters.

**References**

1. Google. (n.d.). _K-means advantages and disadvantages | clustering in machine learning | google developers_. Google. Retrieved April 26, 2022, from [https://developers.google.com/machine-learning/clustering/algorithm/advantages-disadvantages](https://developers.google.com/machine-learning/clustering/algorithm/advantages-disadvantages)
2. Wagstaff, K., Claire Cardie, Rogers, S., &amp;amp; Schroedl, S. (n.d.). _Constrained K-means Clustering with Background Knowledge_. Retrieved April 26, 2022, from [https://cs.cmu.edu/~./dgovinda/pdf/icml-2001.pdf](https://cs.cmu.edu/~./dgovinda/pdf/icml-2001.pdf)
3. Xin Jin, &amp; Jiawei Han. (1970, January 1). _K-means clustering_. SpringerLink. Retrieved April 26, 2022, from [https://link.springer.com/referenceworkentry/10.1007/978-0-387-30164-8\_425](https://link.springer.com/referenceworkentry/10.1007/978-0-387-30164-8_425)
# K Means Clustering Optimization Research

This document explores the k-means clustering algorithm as is pertains to linear algebra. 
In effort to understand potential ways to improve accuracy of this algorithm, research 
was conducted via google colab to measure differences in performance given different approaches.
To summarize some of the research, it was discovered that we could optimize the setup process by 
determining the optimal ammount of clusters in the dataset prior to quarying it. This is done by 
finding the WCSS score prior to centroid classification 

## Background on K-Means Clustering

k-means clustering is a method of vector quantization, originally from signal processing, that aims to partition n observations into k clusters in which each observation belongs to the cluster with the nearest mean, serving as a prototype of the cluster. K - Means is an unsupervised algorithm which means it learns patterns from untaged data. These types 
of algorithms are optimal for market segmentization, document clustering, image segmentization, and 
image compression. 
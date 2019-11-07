# FE 595 SKLearn Assignment
#### Anand Patel
##### I pledge my honor that I have abided by the Stevens Honor System.

## Boston Dataset: Linear Regression

For this problem, we import the Boston dataset from `sklearn` and assign the features and target to variables 
`X` and `Y` respectively. For the features, we drop `CHAS`, which is a dummy variable that indicates whether 
the home is near the Charles River. The target variable represents the median value of a home.

After fitting a linear regression to the model, we can access the `coef_` attribute which tells us the 
coefficients of each predictor. From here, we take the absolute value and print out the predictor name 
that corresponds with the index of the coefficient with the highest absolute value, and thus **we can say
`NOX` (nitrous oxide level in ppm) is the most influential predictor of home price in Boston**.


## Iris Dataset: K-Means Clustering

For the Iris dataset, we are only dealing with the "X" values since we are using an unsupervised model 
in K-Means clustering. To convey the relationship between the number of clusters and the total within-cluster 
distance from the centroids, we will create and fit a K-Means model with `k` ranging from 1 to 10. For each 
model, we apply a function we write to sum the total distances of each point in a cluster to that cluster's 
centroid to each of the clusters in that model.

![Elbow graph](https://github.com/apate107/fe595-sklearn/raw/master/Figure_1.png)

After graphing total distances across the range of `k` values we tested, **we can confirm that there are three 
distinct species** because there is still a significant dropoff in total distance from `k=2` to `k=3` clusters. 
However, for all values of k after, the decrease in distance at each step is decreasing in size.
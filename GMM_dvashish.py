import random
import numpy as np
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

print ( "######## loading data" )
X_train = np.loadtxt ( 'gmm_data.txt' )
print ( "######## data load complete" )

#  Q3.a using sklearn.gmm
gmm = GaussianMixture ( n_components=3, covariance_type='spherical', tol=0.001, init_params='random', random_state=3)
gmm.fit(X_train)

means = gmm.means_
means_sk = gmm.means_
covariances = gmm.covariances_
weights = gmm.weights_
weights_sk = gmm.weights_

labels = gmm.predict(X_train)[:, np.newaxis]
X_train_sk = np.hstack ( (X_train, labels) )
dist_1 = X_train_sk[X_train_sk[:, 5] == 0]
dist_2 = X_train_sk[X_train_sk[:, 5] == 1]
dist_3 = X_train_sk[X_train_sk[:, 5] == 2]

plt.title ( "Feature1 vs Feature2" )
plt.axis ( 'equal' )
plt.scatter ( dist_1[:, 0], dist_1[:, 1], c='red' )
plt.scatter ( dist_2[:, 0], dist_2[:, 1], c='green' )
plt.scatter ( dist_3[:, 0], dist_3[:, 1], c='blue' )
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show ()

plt.title ( "Feature3 vs Feature4" )
plt.axis('equal')
plt.scatter(dist_1[:, 2], dist_1[:, 3], c='red' )
plt.scatter(dist_2[:, 2], dist_2[:, 3], c='green' )
plt.scatter(dist_3[:, 2], dist_3[:, 3], c='blue' )
plt.xlabel("Feature 3")
plt.ylabel("Feature 4")
plt.show()

plt.title("Feature4 vs Feature5")
plt.axis('equal')
plt.scatter(dist_1[:, 3], dist_1[:, 4], c='red')
plt.scatter(dist_2[:, 3], dist_2[:, 4], c='green')
plt.scatter(dist_3[:, 3], dist_3[:, 4], c='blue')
plt.xlabel("Feature 4")
plt.ylabel("Feature 5")
plt.show()

class Model(object):
    def e_step(self, X, n_clusters, weights, means, covariances):
        clusters = []
        for i in range(n_clusters):
            cluster = weights[i] * multivariate_normal.pdf(X, means[i], covariances[i] * np.identity(5))
            clusters.append(cluster.copy())
        clusters = np.asarray(clusters)
        denominator = np.sum(clusters, axis=0)
        return clusters/denominator


    def m_step(self, rnk, X, wts=False):
        means = np.dot(rnk, X)
        denominator = np.sum(rnk, axis=1)[:, np.newaxis]
        means = means / denominator
        if wts:
            weights = np.sum(rnk, axis=1) / np.sum(denominator, axis=0)
            return means, weights
        return means


    def log_likelihood(self, X, n_clusters, rnk, weights, means, covariances):
        log_likes = []
        for i in range(n_clusters):
            log_l = np.log(weights[i]) + multivariate_normal.logpdf(X, means[i], covariances[i] * np.identity(5))
            log_likes.append(log_l.copy())
        log_likes = np.asarray(log_likes)
        log_likelihood = np.sum(np.sum(np.multiply(rnk, log_likes), axis=1), axis=0)
        return log_likelihood

if __name__ == '__main__':
    m = Model()

    # Q3.c means initialized randomly. weights and covariances taken from sklearn.gmm
    random.seed(3)
    means = np.random.rand(3,5)
    rnk = m.e_step(X_train, 3, weights, means, covariances)
    Q_prev = m.log_likelihood ( X_train, 3, rnk, weights, means, covariances )
    means = m.m_step(rnk, X_train)
    diff = 1
    count = 0
    while diff > 0.001:
        rnk = m.e_step(X_train, 3, weights, means, covariances)
        Q_curr = m.log_likelihood ( X_train, 3, rnk, weights, means, covariances )
        means = m.m_step(rnk, X_train)
        diff = Q_curr - Q_prev
        Q_prev = Q_curr
        count = count + 1
    print ( "With only means initialized randomly:" )
    print("Means from EM implementation:")
    print(means)
    print()
    print("Means from sklearn.gmm:")
    print(means_sk)
    print()

    #  Q3.d means and weights initialized randomly. covariances taken from
    random.seed(3)
    weights = np.random.rand(3)
    weights = weights/np.sum(weights)
    means = np.random.rand(3,5)
    rnk = m.e_step(X_train, 3, weights, means, covariances)
    Q_prev = m.log_likelihood(X_train, 3, rnk, weights, means, covariances)
    means, weights = m.m_step(rnk, X_train, wts=True)
    diff = 1
    count = 0
    while diff > 0.001:
        rnk = m.e_step(X_train, 3, weights, means, covariances)
        Q_curr = m.log_likelihood ( X_train, 3, rnk, weights, means, covariances )
        means, weights = m.m_step(rnk, X_train, wts=True)
        diff = Q_curr - Q_prev
        Q_prev = Q_curr
    print("With means and weights initialized randomly:")
    print("Means from EM implementation:")
    print(means)
    print()
    print("Means from sklearn.gmm:")
    print(means_sk)
    print()
    print("Weights from EM implementation:")
    print(weights)
    print()
    print("Weights from slearn.gmm:")
    print(weights_sk)
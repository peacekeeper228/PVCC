from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
w2v_vectors = np.load('lab4/ArticlesVectors.npy')
pca2 = PCA(n_components=2)
pca2.fit(w2v_vectors)
x_3d = pca2.transform(w2v_vectors)
x,y = zip(*x_3d)
plt.scatter(x, y)
plt.show()


from sklearn.metrics import silhouette_score
from sklearn.metrics import adjusted_rand_score
from sklearn.cluster import KMeans
score=[]
for n_clusters in range (2,100):
    clusterer = KMeans(n_clusters=n_clusters).fit(x_3d)
    score.append(silhouette_score(x_3d, clusterer.labels_))
    print("For n_clusters = {}, silhouette score is {}".format(n_clusters, score[n_clusters-2]))

plt.plot(score)
plt.show()
max_value = max(score)
nClusters = score.index(max_value)+2
nClusters = 4
clusterer = KMeans(n_clusters=nClusters).fit(w2v_vectors)
np.save('lab4/ClassificationX', x_3d)
np.save('lab4/ClassificationY', clusterer.labels_)
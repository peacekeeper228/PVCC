from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
import umap
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans

w2v_vectors = np.load('lab4/ArticlesVectors.npy')
#сжатие размерности PCA
pca2 = PCA(n_components=2)
pca2.fit(w2v_vectors)
x_3d = pca2.transform(w2v_vectors)
x,y = zip(*x_3d)
plt.scatter(x, y)
plt.show()
# сжатие размерности UMAP
reducer = umap.UMAP()
embedding = reducer.fit_transform(w2v_vectors)
plt.scatter(
    embedding[:, 0],
    embedding[:, 1])
plt.show()

score=[]
listCluster = [i for i in range(2,21)]
for n_clusters in listCluster:
    clusterer = KMeans(n_clusters=n_clusters).fit(embedding)
    score.append(silhouette_score(embedding, clusterer.labels_))
    print("For n clusters = {}, silhouette score is {}".format(n_clusters, score[n_clusters-2]))

plt.plot(listCluster,score)
plt.xticks(np.arange(2,21,1))
plt.show()
max_value = max(score)
nClusters = score.index(max_value)+2
clusterer = KMeans(n_clusters=nClusters).fit(embedding)
np.save('lab4/ClassificationX', x_3d)
np.save('lab4/ClassificationY', clusterer.labels_)
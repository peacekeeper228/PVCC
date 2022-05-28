from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
import umap
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.cluster import KMeans
from enum import Enum, auto

class CompStatus(Enum):
    PCA = auto()
    UMAP = auto()
class Metric(Enum):
    Silhouette = auto()
    Davies = auto()

choosingComp=CompStatus.UMAP
choosingMetr=Metric.Davies

w2v_vectors = np.load('lab4/ArticlesVectors.npy')
#сжатие размерности PCA
def compessionPCA():
    pca2 = PCA(n_components=2)
    pca2.fit(w2v_vectors)
    x_3d = pca2.transform(w2v_vectors)
    plt.scatter(
        x_3d[:, 0],
        x_3d[:, 1])
    plt.show()
    return x_3d
# сжатие размерности UMAP
def compressionUMAP():
    reducer = umap.UMAP()
    x_3d = reducer.fit_transform(w2v_vectors)
    plt.scatter(
        x_3d[:, 0],
        x_3d[:, 1])
    plt.show()
    return x_3d

if choosingComp==CompStatus.UMAP:
    embedding=compressionUMAP()
else:
    embedding=compessionPCA()

score=[]
listCluster = [i for i in range(2,21)]
for n_clusters in listCluster:
    clusterer = KMeans(n_clusters=n_clusters).fit(embedding)
    if choosingMetr==Metric.Silhouette:
        score.append(silhouette_score(embedding, clusterer.labels_))
    else:
        score.append(davies_bouldin_score(embedding, clusterer.labels_))
    print(f"For n clusters = {n_clusters}, score is {score[n_clusters-2]}")

plt.plot(listCluster,score)
plt.xticks(np.arange(2,21,1))

if choosingMetr==Metric.Davies:
    ekstremum = min(score)
else:
    ekstremum=max(score)
nClusters = score.index(ekstremum)+2
print (nClusters)
clusterer = KMeans(n_clusters=nClusters).fit(embedding)
np.save('lab4/ClassificationX', embedding)
np.save('lab4/ClassificationY', clusterer.labels_)
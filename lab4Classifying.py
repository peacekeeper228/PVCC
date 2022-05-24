import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
arrX = np.load('lab4/ClassificationX.npy')
arrY = np.load('lab4/ClassificationY.npy')
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(arrX, arrY,test_size=0.25)
model = KNeighborsClassifier()
model.fit(X_train,y_train)
res=model.predict(X_test)
def calc_prec(y_hat2, y_test):
    prec=0
    for i, j in zip(y_hat2, y_test):
        if i==j:
            prec+=1
    prec/=len(y_test)
    return prec

print(calc_prec(res, y_test))

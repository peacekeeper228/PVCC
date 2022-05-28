import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import joblib

arrX = np.load('lab4/ClassificationX.npy')
arrY = np.load('lab4/ClassificationY.npy')
X, X_val, Y, Y_val = train_test_split(arrX, arrY,test_size=0.10)
maxAccuracy=0
params={}

def buildModel():
    return KNeighborsClassifier(n_neighbors=11)

for _ in range(100):
    X_train, X_test, y_train, y_test = train_test_split(X, Y,test_size=0.20)
    model = buildModel()
    model.fit(X_train,y_train)
    res=model.predict(X_test)
    accuracy=f1_score(res, y_test,average='micro')
    if accuracy>maxAccuracy:
        maxAccuracy=accuracy
        params=model.get_params()

modelRes=buildModel()
modelRes.set_params(**params)
res=model.predict(X_test)
print('Итоговая доля правильных ответов: {:4.2f}%'.format(accuracy_score(res, y_test)*100))
print('Итоговая точность: {:4.2f}%'.format(precision_score(res, y_test)*100))
print('Итоговая полнота: {:4.2f}%'.format(recall_score(res, y_test)*100))
print('Итоговая метрика f1: {:4.2f}%'.format(f1_score(res, y_test)*100))

joblib_file = "lab4/model.pkl"
joblib.dump(modelRes, joblib_file)
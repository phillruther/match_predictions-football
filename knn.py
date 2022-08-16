import preprocessing
data_path=r"epl_23-f.csv"
trainx,trainy,valx,valy,testx,testy=preprocessing.process(data_path,dt_features=False)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans


kmeans=KMeans(n_clusters=10)
kmeans.fit(trainx,trainy['fthg'])
preds=kmeans.predict(trainx)
print(preds)

def error(preds,y):
    error=(preds==y)/len(preds)
    return error

print(error(preds,trainy['fthg']))
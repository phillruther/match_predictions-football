import preprocessing
import features
import optimization
import utility
data_path=r"epl_23-f.csv"
data=features.RollingFeatures(data_path=data_path).excecute()
# data=preprocessing.postprocess(data)
train,test=utility.splits(data)
train,val=utility.splits(train)
trainx,trainy=features.feature_selection(train)
print(trainx)
print(trainy)





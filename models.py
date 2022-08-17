# regression models
# models i can make
# 1 alinear and higher d regression
# 2 random forest rgressiaon
# 3 svm regression 
# 4 radient descent regression -  i would't do this
from sklearn import linear_model,metrics,ensemble,preprocessing
from sklearn.pipeline import Pipeline



# models for hyperparameter tuning in optimization.py
pipeline_lr=Pipeline([
                    ('scaler',preprocessing.StandardScaler()),
                    ('model',linear_model.LinearRegression())
                    ])

pipeline_rfreg=Pipeline([
                    ('model',ensemble.RandomForestRegressor())
                    ])



# for final training and validation-  for now
class Train:
    def __init__(self,trainx,trainy,valx,valy,testx,testy): #if you  wann apply poly features make a separate instantiation of this class with poly features
        self.trainx=trainx
        self.trainy=trainy
        self.valx=valx
        self.valy=valy
        self.testx=testx
        self.testy=testy
        
    def model_eval(self,model,model_parameters,mode=None,transforms=False):
        model[-1].set_params(**model_parameters)
        if transforms==False:
            model.fit(self.trainx,self.trainy)
        else:
            model.fit_transform(self.trainx,self.trainy)
            
        train_preds=model.predict(self.trainx)
        val_preds=model.predict(self.valx)
        test_preds=model.predict(self.testx)
        if mode=='reg':
            #error score
            train_score,val_score,test_score=metrics.mean_squared_error(self.trainy,train_preds),metrics.mean_squared_error(self.valy,val_preds),metrics.mean_squared_error(self.testy,test_preds)
        elif mode=='cla':
            # accuracay scaore
            train_score,val_score,test_score=metrics.accuracy_score(self.trainy,train_preds),metrics.accuracy_score(self.valy,val_preds),metrics.accuracy_score(self.testy,test_preds)
        return train_score,val_score,test_score 
       
    
    
    
# classification
# 1 logistic  regression
# 2 random forest classifier
# 3 svm classifier
            
        
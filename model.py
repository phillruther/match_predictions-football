# regression models
# models i can make
# 1 alinear and higher d regression
# 2 random forest rgressiaon
# 3 svm regression 
# 4 radient descent regression -  i would't do this
from sklearn import linear_model,metrics,ensemble

class regression:
    def __init__(self,trainx,trainy,valx,valy,testx,testy): #if you  wann apply poly features make a separate instantiation of this class with poly features
        self.trainx=trainx
        self.trainy=trainy
        self.valx=valx
        self.valy=valy
        self.testx=testx
        self.testy=testy
        
    def train_test(self,model,mode=None):
        model.fit(self.trainx,self.trainy)
        train_preds=model.predict(self.trainx,self.trainy)
        val_preds=model.predict(self.valx,self.valay)
        test_preds=self.predict(self.testxa,self.testy)
        if mode=='reg':
            a,b,c= metrics.mean_squared_error(self.trainx,train_preds),metrics.mean_squared_error(self.valx,val_preds),metrics.mean_squared_error(self.testx,test_preds)
        elif mode=='cla':
            a,b,c=metrics.accuracy_score(self.trainx,train_preds),metrics.accuracy_score(self.valx,val_preds),metrics.accuracy_score(self.testx,test_preds)
        return a,b,c 
       
    def linear_regression(self,model_parameters):
        lr=linear_model.LinearRegression.set_params(**model_parameters)
        return self.train_test(model=lr,mode='reg')
    
    def randf_refgression(self,model_parameters):
        rfr= ensemble.RandomForestRegressor.set_params(**model_parameters)
        return self.train_test(model=rfr,mode='reg')
    
    
# classification
# 1 logistic  regression
# 2 random forest classifier
# 3 svm classifier

class classification(regression):
    def logistic_regression(self,model_parameters):
        logr=linear_model.LogisticRegression.set_params(**model_parameters)
        return self.train_test(model=logr,mode='cla')

    def randf_classifier(self,model_parameters):
        rfc= ensemble.RandomForestClassifier.set_params(**model_parameters)
        return self.train_test(model=rfc,mode='cla')
            
        
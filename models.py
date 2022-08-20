# regression models
# models i can make
# 1 alinear and higher d regression
# 2 random forest rgressiaon
# 3 svm regression 
# 4 radient descent regression -  i would't do this
from sklearn import linear_model,metrics,ensemble,preprocessing
from sklearn.pipeline import Pipeline
import numpy as np
import pandas as pd


# models for hyperparameter tuning in optimization.py

pipeline_rfclf=Pipeline([('model',ensemble.RandomForestClassifier())])


# for final training and validation-  for now
class Train:
    def __init__(self,trainx,trainy,valx,testx,s_tr,s_vl,s_ts,model,model_parameters,transforms=False,multi=True): #if you  wann apply poly features make a separate instantiation of this class with poly features
       
        self.trainx=trainx
        self.valx=valx
        self.testx=testx
        self.trainy=trainy
        self.s_tr=s_tr
        self.s_vl=s_vl
        self.s_ts=s_ts
        self.model=model[-1].set_params(**model_parameters)
        self.transforms=transforms
        self.multi=multi
        
    def model_eval(self):
        if self.transforms==False:
            self.model.fit(self.trainx,self.trainy)
        elif self.transforms==True:
            self.model.fit(self.trainx,self.trainy)
            
        train_preds=self.model.predict(self.trainx)
        val_preds=self.model.predict(self.valx)
        test_preds=self.model.predict(self.testx)
        
        if self.multi:
            train_acc=self.multi_out_eval(train_preds,self.s_tr)
            val_acc=self.multi_out_eval(val_preds,self.s_vl)
            test_acc=self.multi_out_eval(test_preds,self.s_ts)
        else:
            train_acc=self.single_out(train_preds,self.trainy)
            val_acc=self.single_out(val_preds,self.valy)
            test_acc=self.single_out(test_preds,self.testy)
                
        return train_acc,val_acc,test_acc
    
    @staticmethod
    def multi_out_eval(preds,real):  
        pred=preds[:,0]-preds[:,1]
        pred=pred/abs(pred)
        pred=pd.DataFrame(pred,columns=['preds'])
        pred['preds'].fillna(0,inplace=True)
        real=real.map({'H':1,'A':-1,'D':0})
        real.reset_index(drop=True,inplace=True)
        acc=(real==pred['preds']).sum()/len(real)
        a=preds
        return acc
    
    @staticmethod
    def single_out(preds,real):
        return metrics.accuracy_score(real,preds)

    
        
from pyexpat import model
from hyperopt import fmin,hp,tpe,Trials
from hyperopt.pyll.base import scope
from sklearn import metrics
from functools import partial
from sklearn.model_selection import KFold,cross_val_score
from sklearn.metrics import confusion_matrix
import preprocessing
import models




# parameters for tuning in models in model.py
randf_clf_params={'n_estimators':scope.int(hp.quniform('n_estimators', 10, 100, 1)), 
               'max_depth':scope.int(hp.quniform('max_depth', 1, 50, 1)),
               'min_samples_split':scope.int(hp.quniform('min_samples_split', 2, 20, 1)),
               'min_samples_leaf':scope.int(hp.quniform('min_samples_leaf', 2, 20, 1)),
               'max_features':hp.quniform('max_features', 0.1, 1, 0.1)}
                                                



class bayes_estimation:
    def  __init__(self,params,trainx,trainy,valx,s_vl,model,transforms):
        self.params=params
        self.trainx=trainx
        self.trainy=trainy
        self.valx=valx
        self.s_vl=s_vl
        self.model=model
        self.transforms=transforms
      
    @staticmethod
    def objective(params,trainx,trainy,valx,s_vl,model,transforms,multi=True): #transforms for model pipelines containing transformations
      
    def optim_on_opptuna(**params):
      return optuna(**params
        model[-1].set_params(**params) #assuming the last method of a pipeline is an estimator
        if transforms==True:
            model.fit(trainx,trainy)
        else:   
            model.fit(trainx,trainy) #because we are giving a pipeline
        preds=model.predict(valx)
        if multi==True:
            out=models.Train.multi_out_eval(preds,s_vl)
        else:
            out=models.Train.single_out(preds,s_vl)
        return out
    
    def parameters(self,multi=True):
        obj_func=partial(self.objective,trainx=self.trainx,trainy=self.trainy,valx=self.valx, s_vl=self.s_vl,model=self.model,transforms=self.transforms,multi=multi)
        trials=Trials()
        result = fmin(fn=obj_func,
                    space=self.params,
                    algo=tpe.suggest,
                    trials=trials,
                    max_evals=10
                        )
        
        return result
    
def format_result(result):
    """formats result paraameters from bayesian search into any desired form- usually a change in data type"""
    result['n_estimators'] = int(result['n_estimators'])
    result['min_samples_split'] = int(result['min_samples_split'])
    result['min_samples_leaf'] = int(result['min_samples_leaf'])
    result['max_depth'] = int(result['max_depth'])

    # result['criterion'] = ['gini', 'entropy'][int(result['criterion'])]
    return result
    

    # def fianl_train(self):    
    #     result=self.parameters()
    #     model=self.model.set_params(**result)
    #     model.fit(self.trainx,self.trainy)
    #     folds=KFold()
    #     train_score=cross_val_score(self.model,self.trainx,self.trainy,scoring='accuracy',cv=folds).mean()
    #     print(train_score)
    #     val_preds=model.predict(self.valx)
    #     return accuracy_score(val_preds,self.valy),confusion_matrix(val_preds,self.valy)
    # 
    # #caustomize this if needed, this is rough

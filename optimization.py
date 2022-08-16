from hyperopt import fmin,hp,tpe,Trials
from hyperopt.pyll.base import scope
from sklearn.metrics import accuracy_score
from functools import partial
from sklearn.model_selection import KFold,cross_val_score
from sklearn.metrics import confusion_matrix
import preprocessing



param_lr={
    
    
    
    }

params_randf = {
    'n_estimators': scope.int(hp.quniform('n_estimators', 10, 100, 1)),
    'max_depth': scope.int(hp.quniform('max_depth', 1, 50, 1)),
    'min_samples_split': scope.int(hp.quniform('min_samples_split', 2, 20, 1)),
    'min_samples_leaf': scope.int(hp.quniform('min_samples_leaf', 2, 20, 1)),
    'max_features': hp.quniform('max_features', 0.1, 1, 0.1),
    'bootstrap': hp.choice('bootstrap', [True, False]),
    'criterion': hp.choice('criterion', ['gini', 'entropy'])}

 # can customize this as well
        
        # result['n_estimators'] = int(result['n_estimators'])
        # result['min_samples_split'] = int(result['min_samples_split'])
        # result['min_samples_leaf'] = int(result['min_samples_leaf'])
        # result['criterion'] = ['gini', 'entropy'][int(result['criterion'])]




class bayes_estimation:
    def  __init__(self,params,trainx,trainy,valx,valy,model):
        self.params=params
        self.trainx=trainx
        self.trainy=trainy
        self.valx=valx
        self.valy=valy
        self.model=model
        
    @staticmethod
    def objective(params,trainx,trainy,valx,valy,model):
        model=model.set_params(**params)
        model.fit(trainx,trainy)
        preds=model.predict(valx)
        acc=accuracy_score(preds,valy)
        return (-1*acc)
    
    def parameters(self):
        obj_func=partial(self.objective,trainx=self.trainx,trainy=self.trainy,valx=self.valx,valy=self.valy,model=self.model)
        trials=Trials()
        result = fmin(fn=obj_func,
                    space=self.params,
                    algo=tpe.suggest,
                    trials=trials,
                    max_evals=10
                        )
        
        return result

    def fianl_train(self):    
        result=self.parameters()
        model=self.model.set_params(**result)
        model.fit(self.trainx,self.trainy)
        folds=KFold()
        train_score=cross_val_score(self.model,self.trainx,self.trainy,scoring='accuracy',cv=folds).mean()
        print(train_score)
        val_preds=model.predict(self.valx)
        return accuracy_score(val_preds,self.valy),confusion_matrix(val_preds,self.valy) #caustomize this if needed, this is rough
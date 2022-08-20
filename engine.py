import warnings
warnings.simplefilter('ignore')
import preprocessing
import features
import optimization
import utility
import models

    
data_path=r"epl_23-f.csv"
data=features.RollingFeatures(data_path=data_path).excecute()
# data=preprocessing.postprocess(data)  #best to only give impuation here rest give it on pipeline in model
train,test=utility.splits(data,test_size=0.2)
train,val=utility.splits(train,test_size=0.2)

trainx,trainy,s1=features.feature_selection(train,multinomial=True)
valx,valy,s2=features.feature_selection(val,multinomial=True)
testx,testy,s3=features.feature_selection(test,multinomial=True)

param_space=optimization.randf_clf_params
model=models.pipeline_rfclf
bayes=optimization.bayes_estimation(param_space,trainx,trainy,valx,s2,model,False)
parameters=bayes.parameters()
parameters=optimization.format_result(parameters)
train=models.Train(trainx,trainy,valx,testx,s1,s2,s3,model,parameters,transforms=False, multi=True)
train_preds,val_preds,test_preds=train.model_eval()

print(train_preds,val_preds,test_preds) 


import warnings
warnings.simplefilter('ignore')
import preprocessing
import features
import optimization
import utility
import models


if __name__ == '__main__':
    
    data_path=r"epl_23-f.csv"
    data=features.RollingFeatures(data_path=data_path).excecute()
    # data=preprocessing.postprocess(data)  #best to only give impuation here rest give it on pipeline in model
    train,test=utility.splits(data)
    train,val=utility.splits(train)
    trainx,trainy=features.feature_selection(train,home=True)
    valx,valy=features.feature_selection(val,home=True)
    testx,testy=features.feature_selection(test,home=True)
    

    params=optimization.randf_reg_params
    model=models.pipeline_rfreg
    bayes=optimization.bayes_estimation(params,trainx,trainy,valx,valy,model)
    result=bayes.parameters()
    result=optimization.format_result(result)
    train_score,val_score,test_score=models.Train(trainx,trainy,valx,valy,testx,testy).model_eval(model=model,model_parameters=result,mode='reg',transforms=False)
    print(train_score,val_score,test_score)



    print(trainx.shape,trainy.shape)
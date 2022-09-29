import warnings
warnings.simplefilter('ignore')
import preprocessing
import features
import optimization
import utility
import models

def app():
      data_path=r"epl_23-f.csv"
      data=features.RollingFeatures(data_path=data_path).excecute()
      # data=preprocessing.postprocess(data)  #best to only give impuation here rest give it on pipeline in model
      train,test=utility.splits(data,test_size=0.2)
      train,val=utility.splits(train,test_size=0.2)

      # trainx,trainy,s1=features.feature_selection(train,multinomial=True)
      # valx,valy,s2=features.feature_selection(val,multinomial=True)
      # testx,testy,s3=features.feature_selection(test,multinomial=True)

      # param_space=optimization.randf_clf_params
      # model=models.pipeline_rfclf
      # bayes=optimization.bayes_estimation(param_space,trainx,trainy,valx,s2,model,False)
      # parameters=bayes.parameters(multi=True)
      # parameters=optimization.format_result(parameters)
      # train=models.Train(trainx,trainy,valx,testx,s1,s2,s3,model,parameters,transforms=False, multi=True)
      # train_preds,val_preds,test_preds=train.model_eval()
      # print(train_preds,val_preds,test_preds) 




      trainx,trainy=features.train_test_split(train)
      valx,valy=features.train_test_split(val)
      testx,testy=features.train_test_split(test)

      param_space=optimization.randf_clf_params
      model=models.pipeline_rfclf
      bayes=optimization.bayes_estimation(param_space,trainx,trainy,valx,valy,model,False)
      parameters=bayes.parameters(multi=False)
      parameters=optimization.format_result(parameters)
      model=model[-1].set_params(**parameters)
      model.fit(trainx,trainy)
      tr_preds=model.predict(trainx)
      vl_preds=model.predict(valx)
      ts_preds=model.predict(testx)

      print(models.Train.single_out(tr_preds,trainy),
            models.Train.single_out(vl_preds,valy),
            models.Train.single_out(ts_preds,testy))

if __name__=='__main__':
      app()

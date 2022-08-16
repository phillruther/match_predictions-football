import preprocessing
import features
import optimization

data_path=r"epl_23-f.csv"
data=features.RollingFeatures(data_path=data_path).excecute()
data=preprocessing.postprocess(data)
features,targets=features.feature_selection(data)


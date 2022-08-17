from sklearn.linear_model import LinearRegression
from statsmodels.tsa.deterministic import DeterministicProcess
import pandas as pd
import preprocessing
import numpy as np
import pandas as pd


# time series features sorta
# trend feature for all of the numerical columns using a dummy dimeansion of 10
def dummy_feature_for_num(data):
    num_data=data.select_dtypes('number')
    dp=DeterministicProcess(
        index=data.date_dummy,
        constant=True,
        order=10,
        drop=True)
    
    for column in num_data.columns:
        if column!='time_dummy':
            y=data[column]
            x=dp.in_sample()
            lr=LinearRegression()
            lr.fit(x,y)
            feature=lr.predict(x)
            string=column+'_tnd'
            data[string]=feature
    return data



# another interesting thing is the motion of avarage variance in any attribute - seems so much rigurous and seems to contain a lot of seasonal changes
# i am gonna treat it as a separate feature and then i'll estimate this feature over time dummy using linear regression
# more like variance trend


def estimate_variance_of_features(data,features,keep_both_features=None):
    """we wil make variance feature first and then we will estimate that feature using regression 
        you can either keep both the variance feature and its estimate or just keep the esrimate as a feature ain the out"""
    var_est=pd.DataFrame()
    var=pd.DataFrame()
    for feature in features:
        var_of_feature=data[feature].rolling(
            window=380, #assuming feature could be having a football seasons season
            center=True,
            min_periods=191,
        ).var()
        string=feature+'_var'
        var[string]=var_of_feature
        dp=DeterministicProcess(
            index=data.date_dummy,
            constant=True,
            order=10,
            drop=True
        )
        x=dp.in_sample()
        y=var_of_feature
        fill=y.mean()
        y.fillna(fill,inplace=True)
        lr=LinearRegression()
        lr.fit(x,y)
        est_var_feature=lr.predict(x)
        string=feature+'_var_'+'tnd'
        var_est[string]=est_var_feature

    var_est.reset_index(drop=True)
    if keep_both_features==True:
        all_faetures=var_est.join(var.reset_index(drop=True))
    else:
        all_faetures=var_est
    return all_faetures


class RollingFeatures:
    """every team  to its current rolling values of features home and away combined"""
    home_features=['season','datetime','date_dummy','hometeam','ftr','htr','hthg','fthg','hs','hst','hc','hf','hy','hr','referee','week','dayofseason','day']
    away_features=['season','datetime','date_dummy','awayteam','ftr','htr','htag','ftag','as','ast','ac','af','ay','ar','referee','week','dayofseason','day']
    general_features=['season','datetime','date_dummy','team','ftr','htr','htg','ftg','s','st','c','f','y','r','referee','week','dayofseason','day']
    def __init__(self,data_path):
        self.data=preprocessing.process_reg(data_path)

    def f_one(self):
        home_data=self.data[self.home_features]
        away_data=self.data[self.away_features]
        home_data.columns=self.general_features
        away_data.columns=self.general_features
        identification=pd.concat([pd.Series(np.ones(home_data.shape[0])),pd.Series(np.zeros(away_data.shape[0]))],axis=0)
        identification.reset_index(drop=True,inplace=True) #identifying home and away teams
        data=pd.concat([home_data,away_data],axis=0) #concatting home aand away 
        data.reset_index(drop=True,inplace=True)
        data['identification']=identification
        data.sort_values(['team','date_dummy'],ascending=True,inplace=True)
        data.reset_index(drop=True,inplace=True)
        return data
        
    def rolling_features(self,data,features):
        """function calculates the rolling mean of the features and returns a dataframe with the rolling mean of the features"""
        rolling_features=pd.DataFrame()
        for feature in features:
            name=feature+'_rol'
            blank_frame=pd.DataFrame()
            blank_frame[name]=data.groupby('team')[feature].rolling(window=38,center=True,min_periods=20).mean().shift(-1).fillna(method='ffill')
            blank_frame.index=blank_frame.index.get_level_values(1)
            rolling_features[name]=blank_frame[name]
        return rolling_features


    def f_two(self,rol_data):
        new_data=self.f_one()
        new_data=new_data.join(rol_data)
        new_data.drop(['htg','ftg','s','st','c','f','y','r'],axis=1,inplace=True)
        n_home_data=new_data[new_data.identification==1]
        n_away_data=new_data[new_data.identification==0]
        n_home_data.drop(['identification'],axis=1,inplace=True)
        n_away_data.drop(['identification'],axis=1,inplace=True)

        n_home_data.columns=list(n_home_data.columns)[:10]+['hthg','fthg','hs','hst','hc','hf','hy','hr']
        n_away_data.columns=list(n_away_data.columns)[:10]+['htag','ftag','as','ast','ac','af','ay','ar']
        n_home_data.rename({'team':'hometeam'},axis=1,inplace=True)
        n_away_data.rename({'team':'awayteam'},axis=1,inplace=True)
        n_home_data.sort_values(['datetime','week','dayofseason','day','ftr','htr','referee'],ascending=True,inplace=True)
        n_away_data.sort_values(['datetime','week','dayofseason','day','ftr','htr','referee'],ascending=True,inplace=True)

        n_home_data.reset_index(drop=True,inplace=True)
        n_away_data.reset_index(drop=True,inplace=True)
        final_data=n_home_data.join(n_away_data[['awayteam','htag','ftag','as','ast','ac','af','ay','ar']])
        return final_data
    

    def excecute(self):
        trans_data=self.f_one()
        rol_data=self.rolling_features(trans_data,features=['htg','ftg','s','st','c','f','y','r'])
        final_data=self.f_two(rol_data)
        return final_data


def feature_selection(data):
    features=data.select_dtypes(include=['float64'])
    features=features.drop(['fthg','ftag'],axis=1) #select variables
    targets=data[['fthg','ftag']]
    return features,targets
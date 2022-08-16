import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
import datetime as dt
from sklearn.model_selection import train_test_split
import numpy as np
# data is that 23 feature one



def season_datetime_fix(data):
    """function creates datetime features. faeatures- a date dummy as if date is starting from begginning of years  
    match week, match day of saeason, 
    Args:
        data (dataframe): data
    """
    data[['week','dayofseason','date_dummy']]= "" 
    data.datetime=pd.to_datetime(data.datetime).dt.date
    for season in data.season.unique():
        season_data=data[data['season']==season]
        start=season_data.iloc[0].datetime
        year=start.year
        delta=start-dt.date(year,1,1)
        season_data['date_dummy']=season_data.datetime-delta
        season_data.week=season_data.date_dummy.apply(lambda x:x.isocalendar()[1])
        season_data.dayofseason=season_data.date_dummy.apply(lambda x: x.timetuple().tm_yday)
        data[data.season==season]=season_data
    data['day']=data['datetime'].apply(lambda x:x.isoweekday())
    return data


def process_reg(data_path,dt_features=None):
    "features added aand rmoved and asplit the data  perticuarly for a regression setup"

    data=pd.read_csv(data_path,encoding='windows-1254')
    data.columns=[i.lower() for i in data.columns]
    # infact i have to drop all all seaons until 2000-01 - caontaains null
    data.drop(data.loc[data['season'].str.contains('1993|1994|1995|1996|1997|1998|1999',regex=True)].index,inplace=True)
    data=season_datetime_fix(data)
    data['week']=data.week.astype(int)
    data['dayofseason']=data.dayofseason.astype(int)
    data['fthg']=data['fthg'].astype(float)
    data['ftag']=data['ftag'].astype(float)
    # oe_result=OrdinalEncoder(categories=[['H','D','A'],['H','D','A']])
    # data[['htr','ftr']]=oe_result.fit_transform(data[['htr','ftr']])
    # oe_team=OrdinalEncoder()
    # data[['hometeam','awayteam']]=oe_team.fit_transform(data[['hometeam','awayteam']])
    # data.drop(['season','referee'],axis=1,inplace=True)
    data.sort_values('date_dummy',inplace=True) 
    # if dt_features==False:
        # data.drop(['datetime','date_dummy'],axis=1,inplace=True)
    # trainx,testx,trainy,testy=train_test_split(data.drop(['fthg','ftag'],axis=1),data[['fthg','ftag']],test_size=0.2)
    # trainx,valx,trainy,valy=train_test_split(trainx,trainy,test_size=0.1)
    # return trainx,trainy,valx,valy,testx,testy
    return data


    
# def drop_leakage(data):
#     data.drop(['ftr','htr','hs','as','hthg','htag','hst','ast','hc','ac','hf','af','hy','ay','hr','ar'])
    



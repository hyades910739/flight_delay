import numpy as np
import pandas as pd
from collections import defaultdict,Counter
from itertools import chain,takewhile,islice
from sklearn.model_selection import train_test_split
from category_encoders import (
    OrdinalEncoder,TargetEncoder,CountEncoder,LeaveOneOutEncoder
)
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

def data_reader(data_path,air_path):
    'read data and joined with airplane dataset'
    data = pd.read_csv(data_path)
    airports = pd.read_csv(air_path,header=None)

    air_header = [
        'id','name','city','country','iata',
        'icao','latitude','longitude','altitude','timezone',
        'dst','tz','type','source'
    ]
    airports.columns = air_header
    airports = airports[
        ['iata','latitude','longitude','altitude','timezone','dst','tz','type']
    ]
    data = data[[
        'flight_no','Week','Arrival','Airline',
        'std_hour','delay_time','flight_date','is_claim'
    ]]
    data['flight_date'] = pd.to_datetime(data['flight_date'])
    data['Airline'] = data['Airline'].fillna('na_')
    data['label'] = (data['is_claim']==800).astype('int')
    data['cancelled'] = (data['delay_time']=='Cancelled').astype('int')
    return data.join(
        airports.rename({'iata':'Arrival'},axis='columns').set_index('Arrival'),
        on='Arrival'
    )    

def splitXY(df,y_col:list):
    x_col = df.columns.drop(y_col)
    return df[x_col],df[y_col]

def add_features(df):
    df = df.copy()
    df['timezone'] = df['timezone'].astype('float')
    df['timezone_diff'] = df.timezone.apply(lambda x: min(abs(x-8),24-abs(x-8)))
    df['year'] = df['flight_date'].dt.year
    df['week_from_2013'] = df['flight_date'].dt.week+ (df['year']-min(df['year']))*52
    df['month'] = df.flight_date.dt.month
    df['month_from_2013'] = df['flight_date'].dt.month+ (df['month']-min(df['month']))*12        
    df['weekday'] = df.flight_date.dt.weekday.astype('object')    
    df['latitude_q10'] = pd.qcut(df['latitude'],10,duplicates='drop').astype(str)
    df['longitude_q10'] = pd.qcut(df['longitude'],10,duplicates='drop').astype(str)
    df['altitude_q10'] = pd.qcut(df['altitude'],10,duplicates='drop').astype(str)
    return df.drop(['flight_date'],1)

def numeric_filter(df): #-> df, df_numeric
    'filter numeric columns, return df w/o numeric cols and df w/'
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    n_df = df.select_dtypes(numerics)
    df = df.drop(n_df.columns,axis=1)
    n_df = n_df.rename({i:i+"_num" for i in df.columns}, axis=1)
    return df,n_df

def upsampling(df,target_rate,label_col='label'): #->np.array
    pos_sel = np.where(df['label']==1)[0]
    neg_sel = np.where(df['label']==0)[0]    
    np_ = len(pos_sel) 
    nn_ = len(neg_sel)
    cur_rate = np_/(np_ + nn_)
    if cur_rate>target_rate:
        return df
    r = (nn_*target_rate-np_)/(np_*(1-target_rate))
    pos_added = np.random.choice(pos_sel,int(r*np_),True)
    return np.concatenate([pos_sel,neg_sel,pos_added])
        

def categ_encoder(df, df_y, cols, encoders=('target','count')): #-> df
    'encode category columns'
    assert len(encoders)>0, 'encoders is empty'        
    df = df[cols]
    fitted_encoders = []
    fitted_df = []
    get_encoder = {
        'target': lambda: TargetEncoder(cols=cols,).fit(df,df_y),
        'count': lambda: CountEncoder(cols=cols,handle_unknown=0,normalize=True).fit(df),
        #'onehot': lambda: OneHotEncoder(cols=cols).fit(df)
    }
    for en_name in encoders:
        encoder = get_encoder[en_name]()
        x = encoder.transform(df)
        x = x.rename({i:i+"_"+en_name for i in x.columns},axis=1)
        fitted_encoders.append(encoder)
        fitted_df.append(x)
    return fitted_df,fitted_encoders,encoders

def evalution(model,x,y):
    conf_mtx = metrics.confusion_matrix(
        y,
        model.predict(x)
    )
    prec = conf_mtx[1,1]/sum(conf_mtx[:,1])
    recall = conf_mtx[1,1]/sum(conf_mtx[1,:])
    acc = (conf_mtx[1,1]+conf_mtx[0,0])/conf_mtx.sum()
    roc = metrics.roc_auc_score(
        y,
        model.predict(x),
    )
    return acc,roc,prec,recall,conf_mtx


    
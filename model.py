from feat_engineer import (
    data_reader,splitXY, add_features, categ_encoder,
    numeric_filter, upsampling
)
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import argparse


class Model(object):
    def __init__(self, use_numeric, upsampling_rate, target_col):
        self.use_numeric = use_numeric
        self.upsampling_rate = upsampling_rate
        self.target_col = target_col
        self._is_fitted = False

    def fit(self, data_path,air_path,sel_feats=None):
        '''
        Args:
        ----
        sel_feats: list of features used in model. you can set this 
                   from features importance list from previous model_log 
                   to remove noise variables.
        '''
        self.air_path = air_path
        data = data_reader(data_path,air_path)
        x,y = splitXY(data,['delay_time','is_claim','label','cancelled'])
        if self.upsampling_rate>0:
            indice = upsampling(y,self.upsampling_rate)
            x = x.iloc[indice]
            y = y.iloc[indice]

        x = add_features(x)
        # get numeric columns
        if self.use_numeric:
            x, x_num = numeric_filter(x)
        fitted_dfs,fitted_encoders,encoder_names = categ_encoder(
            x, y[self.target_col],cols=x.columns, encoders=('target','count')
        )
        self.fitted_encoders = fitted_encoders
        self.encoder_names = encoder_names

        if self.use_numeric:
            fitted_dfs.append(x_num)
        train_feat = pd.concat(fitted_dfs,axis=1) 
        train_label = y[self.target_col]
        rf = RandomForestClassifier(200,max_depth=5)
        if sel_feats:
            rf.fit(train_feat[sel_feats],train_label)       
        else:
            rf.fit(train_feat,train_label)       
        self.model = rf 
        self._is_fitted = True

    def transform(self,data_path,out='label'):
        '''
        data_path: str,A csv file to predict. Must have columns includes:
                    (flight_id,flight_no,Week,Departure,Arrival,Airline,
                     std_hour,delay_time,flight_date,is_claim)
        out: str, either 'label' or 'prob'. whether model should output label(0,1) or proability.
        '''
        assert out in ('label','prob'), "argument out should be either 'label' or 'prob'."
        if not self._is_fitted:
            raise Exception('The model is not fitted yet, fit first.')
        data = data_reader(data_path,self.air_path)
        x,y = splitXY(data,['delay_time','is_claim','label','cancelled'])
        x = add_features(x)
        # get numeric columns
        if self.use_numeric:
            x, x_num = numeric_filter(x)

        test_feats = [en.transform(x) for en in  self.fitted_encoders]
        if self.use_numeric:
            test_feats.append(x_num)
        test_feat = pd.concat(test_feats,axis=1)   
        if out=='label':
            return self.model.predict(test_feat)
        else:
            return self.model.predict_proba(test_feat)

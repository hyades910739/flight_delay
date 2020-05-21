'''
 train/validation model, evaluate it, write records.
'''


import time
import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from feat_engineer import (
    data_reader,splitXY, add_features, categ_encoder,
    evalution, numeric_filter, upsampling
)
from utli import (
    init_logger,eval_formater,write_model_info,
    write_args, parse_args
)
import argparse

SEED = 689
data_path = 'data/flight_delays_data.csv'
air_path = 'data/airports.dat'
logger = init_logger()

if __name__ == "__main__":
    params = parse_args()
    write_args(logger,params)
    use_numeric,upsampling_rate,target_col = params
    
    data = data_reader(data_path,air_path)
    X,Y = splitXY(data,['delay_time','is_claim','label','cancelled'])
    x_train,x_test,y_train,y_test = train_test_split(
        X,Y,test_size=0.1, random_state=SEED
    )
    #upsampling
    if upsampling_rate>0:
        indice = upsampling(y_train,upsampling_rate)
        x_train = x_train.iloc[indice]
        y_train = y_train.iloc[indice]
    #feature engineering:
    logger.writelines('{}: start feature engineering...\n'.format(time.ctime()))
    x_train = add_features(x_train)
    x_test =  add_features(x_test)    
    # get numeric columns
    if use_numeric:
        x_train, x_train_num = numeric_filter(x_train)
        x_test, x_test_num = numeric_filter(x_test)

    fitted_dfs,fitted_encoders,encoder_names = categ_encoder(
        x_train, y_train[target_col],cols=x_train.columns, encoders=('target','count')
    )    
    if use_numeric:
        fitted_dfs.append(x_train_num)

    train_feat = pd.concat(fitted_dfs,axis=1) 
    train_label = y_train[target_col]
    #modeling:
    logger.writelines('{}: start modeling...\n'.format(time.ctime()))
    rf = RandomForestClassifier(200,max_depth=5)
    rf.fit(train_feat,train_label)
    #validation:
    logger.writelines('{}: start evaluation...\n'.format(time.ctime()))
    test_feats = [en.transform(x_test) for en in  fitted_encoders]
    if use_numeric:
        test_feats.append(x_test_num)
    test_feat = pd.concat(test_feats,axis=1)
    test_label = y_test[target_col]
    train_res = eval_formater(evalution(rf,train_feat,train_label))
    test_res = eval_formater(evalution(rf,test_feat,test_label))
    logger.writelines('{}: evaluation done.\n'.format(time.ctime()))
    write_model_info(logger,rf,train_feat)
    logger.writelines('train:\n----\n')
    logger.writelines(train_res)
    logger.writelines('validation:\n----\n')
    logger.writelines(test_res)
    logger.close()
    print('completed!')



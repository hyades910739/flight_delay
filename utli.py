from math import pi,sqrt,sin,cos,atan2
from category_encoders import (
    OrdinalEncoder,TargetEncoder,CountEncoder,LeaveOneOutEncoder,OneHotEncoder
)
import time
import os
import argparse

def haversine(pos1, pos2):
    # source: https://stackoverflow.com/a/18144531
    lat1 = float(pos1['lat'])
    long1 = float(pos1['long'])
    lat2 = float(pos2['lat'])
    long2 = float(pos2['long'])

    degree_to_rad = float(pi / 180.0)

    d_lat = (lat2 - lat1) * degree_to_rad
    d_long = (long2 - long1) * degree_to_rad

    a = pow(sin(d_lat / 2), 2) + cos(lat1 * degree_to_rad) * cos(lat2 * degree_to_rad) * pow(sin(d_long / 2), 2)
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    km = 6367 * c
    return km

def init_logger():
    name = time.strftime('modellog_%Y%m%d_%H%M')
    name = os.path.join('model_history/',name)
    f = open(name,'wt')
    f.writelines("{} : Logger init\n".format(time.ctime()))
    return f

def write_model_info(f,model,x,writeFI=False):
    'writeFI: whether to write Feature importtance as another file'
    f.writelines('model info:\n----\n')
    f.writelines('model paras: {}\n'.format(model.get_params()))
    f.writelines('**feature_importances**:\n')
    iter_ = zip(x.columns,model.feature_importances_)    
    for i,j in sorted(iter_,key=lambda x:x[1],reverse=True):
        f.writelines('{:<15}: {}\n'.format(i,j))    
    f.writelines('**feature_importances**\n')

def write_args(f,params):
    f.writelines('parameters:\n')
    f.writelines('  use_numeric: {}\n'.format(params[0]))
    f.writelines('  upsampling_rate: {}\n'.format(params[1]))
    f.writelines('  target_col: {}\n'.format(params[2]))

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-u', help='upsampling rate on positive items',default=0.3)
    parser.add_argument('-n', help='whether use numeric_cols in model(and treat as numeric)',default=True)
    parser.add_argument('-target', help='y target to predict,default label.',default='label')
    args = parser.parse_args()
    use_numeric = bool(args.n)
    upsampling_rate = float(args.u)
    target_col = args.target
    return use_numeric,upsampling_rate,target_col



def eval_formater(res): #->str
    return 'accuracy  : {x[0]}\nROC_auc   : {x[1]}\nprecision : {x[2]}\nrecall    : {x[3]}\nconfusion matrix:\n{x[4]}\n'.format(x=res)
import pickle
from utli import parse_args
import time
import os
from model import Model

data_path = 'data/flight_delays_data.csv'
air_path = 'data/airports.dat'

def save_model(model):
    name = os.path.join('models',time.strftime('model.%Y%m%d_%H%M')) 
    with open(name,'wb') as f:
        pickle.dump(model,f)

if __name__ == "__main__":
    use_numeric,upsampling_rate,target_col = parse_args()
    model = Model(use_numeric,upsampling_rate,target_col)
    model.fit(data_path, air_path, sel_feats=None)
    save_model(model)
    
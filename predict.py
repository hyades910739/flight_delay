'''
predict test data by model
'''

from model import Model
import pickle
import time
import argparse

model_path = 'models/model.20200520_1410'

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-out', help="either 'label' or 'prob'. whether model should output label(0,1) or class proability",default='label')
    parser.add_argument('-test_path', help='the path for test_file',required=True)
    args = parser.parse_args()
    return args.out, args.test_path

def load_model(path):
    with open(path,'rb') as f:
        model = pickle.load(f)
    return model

def write_result(pred):
    name = time.strftime('predict_%Y%m%d_%H%M')
    with open(name,'wt') as f:
        for i in pred:
            f.writelines('{}\n'.format(i))

if __name__ == "__main__":
    out,test_path = parse_args()
    model = load_model(model_path)
    pred = model.transform(test_path, out)
    write_result(pred)



import os

def load(dataset_name):
    dataset_path = f'datasets/{dataset_name}'
    with open(dataset_path if os.path.isfile(dataset_path) else dataset_path+'.txt' , 'r') as f:
        data = f.read().split('\n')
    return list(filter(lambda e: e is not "", data))

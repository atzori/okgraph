import os


def load(dataset_name):
    """
    Load dataset from datasets folder by filename.
    :param dataset_name: the dataset filename.
    :return: the list of data
    """
    dataset_path = f'datasets/{dataset_name}'
    with open(dataset_path if os.path.isfile(dataset_path) else dataset_path+'.txt', 'r') as f:
        data = f.read().split('\n')
    return list(filter(lambda e: e is not "", data))

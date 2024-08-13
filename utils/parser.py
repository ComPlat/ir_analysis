import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset

class IrDataset(Dataset):

    ''' Data Loader class for Pytorch DataLoader
    '''

    def __init__(self, df, num_classes, transform=None):
        self.df = df
        self.transform = transform
        self.num_classes = num_classes

    def __len__(self):
        return self.df.shape[0]
        
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        xs = self.df.iloc[idx]['spectrum'].reshape(1, -1)
        ys = np.array([i for i in self.df.iloc[idx]['concat_label']]).astype('float').reshape(-1)
        xs = torch.from_numpy(xs).float()
        ys = torch.from_numpy(ys).float()

        sample = { 
            'xs': xs,
            'ys': ys 
        }

        if self.transform:
            sample = self.transform(sample)
        return sample


def concat_label(dataset):

    ''' Concat string binary labels
    Args:
        dataset : pandas dataframe
    Returns:
        dataset : pandas dataframe with additional column 

    '''

    labels = dataset[dataset.columns[4:]]
    dataset['total_func'] = labels.sum(axis=1).astype(str)
    dataset['concat_label'] = labels.astype(int).astype(str).apply(''.join, axis=1).astype(str)
    dataset.reset_index(drop=True)

    return dataset


def get_dataset(path1, mix, **kwargs):  
            
    ''' Get dataset dataframe suited for dataloader
    Args:
        path1 : dataset pickle file path
        mix : If the dataset is mixed (Boolean)
        path2 : dataset pickle file path to mix

    Returns:
        dataset : pandas dataframe

    '''
    if mix == True :
        if len(kwargs['path2']) > 0 :
            dataset_1 = pd.read_pickle(path1)
            dataset_2 = pd.read_pickle(kwargs['path2'])
            dataset = pd.concat([dataset_1, dataset_2], ignore_index=True)
        else:
            raise FileNotFoundError('Mention path2 if mix = True')
    elif mix == False :
        dataset = pd.read_pickle(path1)
    
    dataset = concat_label(dataset)
    return dataset

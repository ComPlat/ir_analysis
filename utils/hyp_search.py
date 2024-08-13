import os
from pathlib import Path
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold, ParameterGrid
import numpy as np

from parser import IrDataset, get_dataset
from utils import convert_to_binary
from utils import molecule_f1
import warnings
warnings.filterwarnings("ignore", category=UserWarning) 
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

ROOT_DIR = Path(__file__).parent.absolute().parent

'''
This code takes input file from the environment variables 
export DF=/path/to/input/file

'''

## Experiment name
exp_name = 'fc_hyp_search'

## Number of classes
num_classes = 17

## Logging after 
log_interval = 2

model_path = os.path.join(ROOT_DIR, 'models') 

Path(os.path.join(model_path, exp_name)).mkdir(parents=True, exist_ok=True)
model_path = os.path.join(model_path, exp_name)

results_path = os.path.join(ROOT_DIR, 'results')
Path(os.path.join(results_path, exp_name)).mkdir(parents=True, exist_ok=True)
results_path = os.path.join(results_path, exp_name) 

main_df = os.environ['DF']
dataset = get_dataset(main_df, False)
emb_ds = dataset[['cano_smi','concat_label', 'spectrum']]

kf = KFold(n_splits=5, shuffle=True, random_state=42)

## Hyperparameter search space

param_grid = {
    'hidden_size': [64, 128],
    'num_hidden_layers': [1, 2],
    'dropout_rate': [0.2, 0.3],
    'learning_rate': [0.001, 0.01, 0.1],
    'batch_size': [32, 64],
    'num_epochs': [50,100]
}

class FullyConnectedNN(nn.Module):
    '''
    Fully connected network class definition to dynamically 
    configure architecture based on hyper param search space.

    '''
    def __init__(self, input_size, hidden_size, num_hidden_layers, dropout_rate, output_size):
        super(FullyConnectedNN, self).__init__()
        self.hidden_layers = nn.ModuleList()
        self.dropout = nn.Dropout(dropout_rate)

        self.hidden_layers.append(nn.Linear(input_size, hidden_size))
        self.hidden_layers.append(nn.ReLU())
        self.hidden_layers.append(self.dropout)

        for _ in range(num_hidden_layers - 1):
            self.hidden_layers.append(nn.Linear(hidden_size, hidden_size))
            self.hidden_layers.append(nn.ReLU())
            self.hidden_layers.append(self.dropout)

        self.output_layer = nn.Linear(hidden_size, output_size)
        

    def forward(self, x):
        for layer in self.hidden_layers:
            x = layer(x)
        x = self.output_layer(x)

        return torch.sigmoid(x)


def train(model, optimizer, criterion, train_loader, device):
    
    ''' Traning function for neural network

    Args:
        model : Initialized neural network model object
        optimizer : Optimizer object
        scheduler : Learning rate scheduler object
        criterion : Loss function object
        train_loader : Pytorch DataLoader object with training dataset
        device : Device to deploy neural network (cpu/cuda)

    Returns:
        mean_batch_loss : mean batch loss
        mean_batch_f1 : mean batch f1 score
        model : trained model

    '''
    batch_loss = []
    batch_f1 = []
    model.train()
    for batch_idx, data in enumerate(train_loader):
        xs, ys = data['xs'].to(device), data['ys'].to(device)

        optimizer.zero_grad()
        output = model(xs)
        output = output.squeeze(1)
        loss = criterion(output, ys)
        
        loss = (loss).mean()
        binary_preds = convert_to_binary(output.cpu(), torch.tensor([0.5]))

        f1_score = molecule_f1(np.array(binary_preds), ys.cpu())
        batch_loss.append(loss.detach().cpu().numpy())

        batch_f1.append(f1_score)
        
        loss.backward()
        optimizer.step()
    
    mean_batch_f1 = np.mean([x for x in batch_f1], axis=0)
    mean_batch_loss =  np.array(batch_loss).mean()
    return mean_batch_loss, mean_batch_f1, model

def test(model, epoch, total_epochs, criterion, test_loader, device):

    ''' Testing function for neural network

    Args:
        model : Trained neural network model object
        epoch : Current epoch 
        total_epochs : Total epochs
        criterion : Loss function object
        test_loader : Pytorch DataLoader object with testing dataset
        device : Device to deploy neural network (cpu/cuda)

    Returns:
        epoch_loss : mean epoch loss
        test_f1 : f1 scores for test set
        preds : output binary predictions and actual labels
        
    '''
        
    model.eval()
    test_loss = []
    test_f1 = []
    preds = []
    for data in test_loader:
        xs, ys = data['xs'].to(device), data['ys'].to(device)

        predictions = model(xs)
        output = predictions.squeeze(1)
        binary_preds = convert_to_binary(output.cpu(), torch.tensor([0.5]))
        
        mol_f1 = molecule_f1(np.array(binary_preds), ys.cpu())
        
        loss = criterion(output, ys)

        loss = (loss).mean()

        if epoch == (total_epochs - 1):
            preds.append((binary_preds[0], ys.cpu()))
        test_loss.append(loss.detach().cpu().numpy())

        test_f1.append(mol_f1)

    epoch_loss = np.array(test_loss).mean()

    return epoch_loss, test_f1, preds


def train_and_evaluate_model(model, criterion, optimizer, train_loader, val_loader, num_epochs, fold_number, device, idx):

    fold_train_loss = []
    fold_train_f1 = []
    fold_test_loss = []
    fold_test_f1 = []
    plot_logs = []
    
    start_time = time.time()

    for epoch in range(num_epochs):
        
        train_loss, train_f1, model= train(model, optimizer, criterion, train_loader, device)
        test_loss, test_f1, preds = test(model, epoch, num_epochs, criterion, val_loader, device)
           
        fold_train_f1.append(train_f1)
        fold_train_loss.append(train_loss)
        fold_test_f1.append(test_f1)
        fold_test_loss.append(test_loss)

        plot_logs.append([epoch, fold_number,train_f1, train_loss, test_f1, test_loss])
        torch.save(model, model_path+'/{}_{}.pt'.format(exp_name, fold_number))


    with open(results_path+'/log_file_{}.txt'.format(exp_name), 'a+') as file:
        file.write('{} fold number {} : tr_loss :{}, tr_f1 :{}, ts_loss :{}, ts_f1 :{}'.format(
            exp_name, fold_number, train_loss, train_f1, test_loss, test_f1
        ))
        file.write('\n')


    np.save(results_path+'/{}_{}_tr_f1'.format(idx,fold_number), fold_train_f1)
    np.save(results_path+'/{}_{}_tr_loss'.format(idx,fold_number), fold_train_loss)
    np.save(results_path+'/{}_{}_ts_f1'.format(idx,fold_number), fold_test_f1)
    np.save(results_path+'/{}_{}_ts_loss'.format(idx,fold_number), fold_test_loss)
    np.save(results_path+'/{}_{}_plot_logs'.format(idx,fold_number), plot_logs)

    np.save(results_path+'/{}_{}_preds'.format(idx,fold_number), preds)


    print("--- %s seconds ---" % (time.time() - start_time))

    return test_loss, test_f1


best_f1 = np.zeros(17)
best_params = {}
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

all_f1s = []
params_dict ={}
param_f1_dict = {}

for idx, params in enumerate(ParameterGrid(param_grid)):
    print(f"Chosen parameters: {params}")
    params_dict[idx] = params
    
    avg_f1 = []

    for fold, (train_index, val_index) in enumerate(kf.split(dataset)):

        train_df, test_df = emb_ds.loc[train_index], emb_ds.loc[val_index]

        train_dataset = IrDataset(
            df=train_df,
            num_classes=num_classes
        )


        test_dataset = IrDataset(
            df=test_df,
            num_classes=num_classes
        )

        train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True, num_workers=0)
        val_loader = DataLoader(test_dataset, batch_size=len(test_dataset) , shuffle=True, num_workers=0)

    
        model = FullyConnectedNN(input_size=3400, hidden_size=params['hidden_size'],
                                 num_hidden_layers=params['num_hidden_layers'], dropout_rate=params['dropout_rate'],
                                 output_size=num_classes).to(device)
        criterion = nn.BCELoss(reduction='none')
        optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'])
        


        val_loss, val_f1 = train_and_evaluate_model(model, criterion, optimizer, train_loader, val_loader,
                                                          params['num_epochs'], fold, device, idx)
        
        
        


        avg_f1.append(val_f1[0])

    avg_f1_per_param = np.mean(avg_f1, axis= 0)
    param_f1_dict[idx] = avg_f1_per_param
    print(f"Average f1 for these parameters: {avg_f1_per_param}\n")



np.save(results_path+'/params_dict', params_dict)
np.save(results_path+'/param_f1_dict', param_f1_dict)

print('search completed...')

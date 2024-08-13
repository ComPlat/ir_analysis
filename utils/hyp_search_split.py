import warnings
import os
from pathlib import Path
import time

from sklearn.model_selection import KFold, ParameterGrid
import numpy as np

from parser import IrDataset, get_dataset
from utils import convert_to_binary
from utils import molecule_f1

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LinearLR

warnings.filterwarnings("ignore", category=UserWarning) 
ROOT_DIR = Path(__file__).parent.absolute().parent
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

'''
This code takes input file from the environment variables 
export DF=/path/to/input/file

'''
## Experiment name
exp_name = 'Split_search_bnorm_new'

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
dataset = dataset.reset_index(drop=True)
emb_ds = dataset[['cano_smi','concat_label', 'spectrum']]

# Initialize the KFold object for 5-fold cross-validation
kf = KFold(n_splits=5, shuffle=True)
## Hyperparameter search space
param_grid = {
     'learning_rate': [0.01,0.01],
     'batch_size': [32],
     'num_epochs': [50, 100],
     'split_at': [1000, 1200, 1500],
     'num_hidden_layers':[1, 2],
     'hidden_layer_size' :[128, 256],
     'dropout_rate1':[0.2],
     'dropout_rate2':[0.3],
}

class FingerPrint(nn.Module):

    ''' Create Fingerprint neural network class object
    Args:
        inp : input fingerprint data slice of spectrum
    Returns:
        output : sigmoid output of the neural network
    '''

    def __init__(self, inp, num_hidden_layers, hidden_layer_size, dropout_rate1, dropout_rate2):
        super().__init__()

        layers = []
        layers.append(nn.Linear(inp, inp))
        layers.append(nn.BatchNorm1d(inp))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Dropout(dropout_rate1))
        
        for _ in range(num_hidden_layers):
            if _ == 0 :     
                layers.append(nn.Linear(inp, hidden_layer_size))
            else : 
                layers.append(nn.Linear(hidden_layer_size, hidden_layer_size))
            layers.append(nn.BatchNorm1d(hidden_layer_size))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout_rate2))

        self.finger_layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.finger_layers(x.squeeze(1))
        x = torch.flatten(x, 1)
        return x

class SpectraModel(nn.Module):

    ''' Create neural network class object
    Args:
        num_classes : number of classes,
        split_at : wavenumber to split spectrum at
    Returns:
        output : sigmoid output of the neural network
    '''


    def __init__(self, input_size, num_classes, split_at, num_hidden_layers_fc, hidden_layer_size_fc, num_hidden_layers_finger, hidden_layer_size_finger, dropout_rate1, dropout_rate2):
        super().__init__()

        diff = input_size - split_at

        fc_layers = []
        fc_layers.append(nn.Linear(diff, diff))
        fc_layers.append(nn.BatchNorm1d(diff))
        fc_layers.append(nn.ReLU(inplace=True))
        fc_layers.append(nn.Dropout(dropout_rate1))
        
        for _ in range(num_hidden_layers_fc):
            if _ == 0:
                fc_layers.append(nn.Linear(diff, hidden_layer_size_fc))
            else:
                fc_layers.append(nn.Linear(hidden_layer_size_fc, hidden_layer_size_fc))
            fc_layers.append(nn.BatchNorm1d(hidden_layer_size_fc))
            fc_layers.append(nn.ReLU(inplace=True))
            fc_layers.append(nn.Dropout(dropout_rate2))

        self.fc_layers = nn.Sequential(*fc_layers)
        
        self.finger = FingerPrint(split_at, num_hidden_layers_finger, hidden_layer_size_finger, dropout_rate1, dropout_rate2)
        self.fc5 = nn.Linear(hidden_layer_size_fc + hidden_layer_size_finger, num_classes)  

    def forward(self, x_func, x_fing): 
        x_func = self.fc_layers(x_func.squeeze(1))
        x_func = torch.flatten(x_func, 1)

        x = torch.cat([x_func, self.finger(x_fing)], axis=1)
        x = self.fc5(x)
        output = torch.sigmoid(x)
        return output


def train(model, optimizer, criterion, scheduler, train_loader, split_at,  device):

    ''' Traning function for neural network

    Args:
        model : Initialized neural network model object
        optimizer : Optimizer object
        scheduler : Learning rate scheduler object
        criterion : Loss function object
        train_loader : Pytorch DataLoader object with training dataset
        split_at : wavenumber to split spectrum at
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
        x_func, x_fing  = xs[:,:,split_at:].to(device), xs[:,:,:split_at].to(device)

        optimizer.zero_grad()
        output = model(x_func, x_fing)
        #output = output.squeeze(1)
        loss = criterion(output, ys)
        
        loss = (loss).mean()
        binary_preds = convert_to_binary(output.cpu(), torch.tensor([0.5]))

        f1_score = molecule_f1(np.array(binary_preds), ys.cpu())
        batch_loss.append(loss.detach().cpu().numpy())

        batch_f1.append(f1_score)
        
        loss.backward()
        optimizer.step()
        scheduler.step()
    
    mean_batch_f1 = np.mean([x for x in batch_f1], axis=0)
    mean_batch_loss = np.array(batch_loss).mean()
    return mean_batch_loss, mean_batch_f1, model

def test(model, epoch, total_epochs, criterion, test_loader, split_at, device):

    ''' Testing function for neural network

    Args:
        model : Trained neural network model object
        epoch : Current epoch 
        total_epochs : Total epochs
        criterion : Loss function object
        test_loader : Pytorch DataLoader object with testing dataset
        split_at : wavenumber to split spectrum at
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
    with torch.no_grad():
        for data in test_loader:
            xs, ys = data['xs'].to(device), data['ys'].to(device)
            x_func, x_fing  = xs[:,:,split_at:].to(device), xs[:,:,:split_at].to(device)

            predictions = model(x_func, x_fing)
            
            #output = predictions.squeeze(1)
            binary_preds = convert_to_binary(predictions.cpu(), torch.tensor([0.5]))
        
            mol_f1 = molecule_f1(np.array(binary_preds), ys.cpu())
        
            loss = criterion(predictions, ys)

            loss = (loss).mean()

            if epoch == (total_epochs - 1):
                preds.append((binary_preds[0], ys.cpu()))
            test_loss.append(loss.detach().cpu().numpy())

            test_f1.append(mol_f1)

    epoch_loss = np.array(test_loss).mean()

    return epoch_loss, test_f1, preds

# Function to train and evaluate the model
def train_and_evaluate_model(model, criterion, optimizer, scheduler, train_loader, val_loader, num_epochs, split_at, fold_number, device, idx):

    fold_train_loss = []
    fold_train_f1 = []
    fold_test_loss = []
    fold_test_f1 = []
    plot_logs = []
    
    start_time = time.time()

    for epoch in range(num_epochs):
        
        train_loss, train_f1, model= train(model, optimizer, criterion, scheduler, train_loader, split_at, device)
        test_loss, test_f1, preds = test(model, epoch, num_epochs, criterion, val_loader, split_at, device)
        
        # if epoch % log_interval == 0:
        #     print('{} fold number {} : tr_loss :{}, tr_f1 :{}, ts_loss :{}, ts_f1 :{}'.format(
        #     epoch, fold_number, train_loss, train_f1, test_loss, test_f1))
        
        fold_train_f1.append(train_f1)
        fold_train_loss.append(train_loss)
        fold_test_f1.append(test_f1)
        fold_test_loss.append(test_loss)
        plot_logs.append([epoch, fold_number,train_f1, train_loss, test_f1, test_loss])
        scheduler.step()
    
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

# Perform the hyperparameter search using cross-validation
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
        model = SpectraModel(input_size=3400, num_classes=17, split_at=params['split_at'], num_hidden_layers_fc=params['num_hidden_layers'], hidden_layer_size_fc=params['hidden_layer_size'],num_hidden_layers_finger=params['num_hidden_layers'], hidden_layer_size_finger=params['hidden_layer_size'], dropout_rate1=params['dropout_rate1'], dropout_rate2=params['dropout_rate2']).to(device)
        criterion = nn.BCELoss(reduction='none')
        optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'])
        scheduler = LinearLR(optimizer, start_factor=0.5, total_iters=100)
        val_loss, val_f1 = train_and_evaluate_model(model, criterion, optimizer, scheduler, train_loader, val_loader,
                                                          params['num_epochs'],params['split_at'], fold, device, idx)
        
        
        
        avg_f1.append(val_f1[0])

    avg_f1_per_param = np.mean(avg_f1, axis= 0)
    param_f1_dict[idx] = avg_f1_per_param
    print(f"Average f1 for these parameters: {avg_f1_per_param}\n")



np.save(results_path+'/params_dict', params_dict)
np.save(results_path+'/scores_f1_dict', param_f1_dict)

print('search completed...')

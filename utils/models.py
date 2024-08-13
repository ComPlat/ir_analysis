import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
from torch.optim.lr_scheduler import LinearLR

from utils import convert_to_binary
from utils import group_accuracy
from utils import pred_acc


# Split Spectra Neural network definition
class FingerPrint(nn.Module):

    ''' Create Fingerprint neural network class object
    Args:
        inp : input fingerprint data slice of spectrum
    Returns:
        output : sigmoid output of the neural network
    '''

    def __init__(self, inp):
        super().__init__()

        self.fc1 = nn.Linear(inp, inp)
        self.fc1_bn = nn.BatchNorm1d(inp)
        self.dropout_fc1 = nn.Dropout(0.2)
        
        self.fc2 = nn.Linear(inp, 256)
        self.fc2_bn = nn.BatchNorm1d(256)
        self.dropout_fc2 = nn.Dropout(0.3)

    def forward(self, x) : 

        x = F.relu(self.fc1_bn(self.fc1(x.squeeze(1))))
        x = self.dropout_fc1(x)
        x = F.relu(self.fc2_bn(self.fc2(x)))
        x = self.dropout_fc2(x)
        x = torch.flatten(x,1)

        return x

class SpectraModel(nn.Module):
    
    ''' Create neural network class object
    Args:
        num_classes : number of classes,
        split_at : wavenumber to split spectrum at
    Returns:
        output : sigmoid output of the neural network
    '''

    def __init__(self, num_classes, split_at):
        
        super().__init__()

        diff = 3400 - split_at

        self.fc1 = nn.Linear(diff, diff) 
        self.fc1_bn = nn.BatchNorm1d(diff)
        self.dropout_fc1 = nn.Dropout(0.2)
        
        self.fc2 = nn.Linear(diff, 256)
        self.fc2_bn = nn.BatchNorm1d(256)
        self.dropout_fc2 = nn.Dropout(0.3)

        self.finger = FingerPrint(split_at)
        self.fc5 = nn.Linear(512, num_classes)  

    def forward(self, x_func, x_fing) : 

        x_func = F.relu(self.fc1_bn(self.fc1(x_func.squeeze(1))))
        x_func = self.dropout_fc1(x_func)
        x_func = F.relu(self.fc2_bn(self.fc2(x_func)))
        x_func = self.dropout_fc2(x_func)
        x_func = torch.flatten(x_func,1)

        x = torch.cat([x_func, self.finger(x_fing)], axis= 1)
        x = self.fc5(x)
        output = torch.sigmoid(x)

        return output


def model_init(lr, num_classes, split_at):

    ''' Initialize Neural network object
    Args:
        lr : learning rate
        num_classes : number of classes,
        split_at : wavenumber to split spectrum at
    Returns:
        model : Initialized neural network model object
        optimizer : Optimizer object
        criterion : Loss function object 
        scheduler : Learning rate scheduler object 
        device : Device to deploy neural network (cpu/cuda)
    '''

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SpectraModel(num_classes, split_at).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss(reduction='none')
    scheduler = LinearLR(optimizer, start_factor=0.5, total_iters=100)
    return model, optimizer, criterion, scheduler, device


def train(model, optimizer, scheduler, criterion, train_loader, device, split_at):

    ''' Traning function for neural network

    Args:
        model : Initialized neural network model object
        optimizer : Optimizer object
        scheduler : Learning rate scheduler object
        criterion : Loss function object
        train_loader : Pytorch DataLoader object with training dataset
        device : Device to deploy neural network (cpu/cuda)
        split_at : wavenumber to split spectrum at

    Returns:
        loss : mean batch loss
        accuracy_1 : mean batch accuracy of predicting 1
        accuracy_0 : mean batch accuracy of predicting 0
        model : trained model

    '''

    batch_loss = []
    batch_accuracy = []
    model.train()
    for batch_idx, data in enumerate(train_loader):
        xs, ys = data['xs'], data['ys'].to(device)
        x_func, x_fing  = xs[:,:,split_at:].to(device), xs[:,:,:split_at].to(device)

        optimizer.zero_grad()
        output = model(x_func, x_fing)
        loss = criterion(output, ys)
        
        loss = (loss).mean()
        binary_preds = convert_to_binary(output.detach().cpu(), torch.tensor([0.5]))
        acc_1, acc_0 = group_accuracy(binary_preds, ys.detach().cpu(), agg='mean')
        batch_loss.append(loss.detach().cpu().numpy())

        batch_accuracy.append((acc_1, acc_0))
        
        loss.backward()
        optimizer.step()
        scheduler.step()
    
    mean_batch_acc1 = np.mean([x[0] for x in batch_accuracy])
    mean_batch_acc0 = np.mean([x[1] for x in batch_accuracy])

    return np.array(batch_loss).mean(), mean_batch_acc1, mean_batch_acc0, model

def test(model, epoch, total_epochs, criterion, test_loader, device, split_at):

    ''' Testing function for neural network

    Args:
        model : Trained neural network model object
        epoch : Current epoch 
        total_epochs : Total epochs
        criterion : Loss function object
        test_loader : Pytorch DataLoader object with testing dataset
        device : Device to deploy neural network (cpu/cuda)
        split_at : wavenumber to split spectrum at

    Returns:
        loss : mean epoch loss
        accuracy_1 : mean epoch accuracy of predicting 1
        accuracy_0 : mean epoch accuracy of predicting 0
        b_preds : binary predictions array for test set
        raw_preds : output logits
        
    '''

    model.eval()
    test_loss = []
    test_accuracy = []
    b_preds = []
    raw_preds = []
    with torch.no_grad():
        for data in test_loader:
            xs, ys = data['xs'], data['ys'].to(device)
            x_func, x_fing  = xs[:,:,split_at:].to(device), xs[:,:,:split_at].to(device)

            predictions = model(x_func, x_fing)
            binary_preds = convert_to_binary(predictions.cpu(), torch.tensor([0.5]))
            acc_1, acc_0 = pred_acc(binary_preds[0], ys.cpu())
            
            loss = criterion(predictions, ys)

            loss = (loss).mean()

            if epoch == (total_epochs - 1):
                b_preds.append((binary_preds[0], ys.cpu()))
                raw_preds.append(predictions.cpu(), ys.cpu())
            test_loss.append(loss.cpu())
            test_accuracy.append((acc_1, acc_0))

    epoch_loss = np.array(test_loss).mean()
    epoch_acc1 = np.mean([j[0] for j in test_accuracy])
    epoch_acc0 = np.mean([j[1] for j in test_accuracy])

    return epoch_loss, epoch_acc1, epoch_acc0, b_preds, raw_preds





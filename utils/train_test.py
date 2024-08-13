import os
from pathlib import Path
import time
import argparse
import json

from sklearn.model_selection import KFold
import numpy as np 

from torch.utils.data import DataLoader
import torch

from models import model_init, train, test
from parser import IrDataset, get_dataset
from reports import generate_report, plot_f1_scores

ROOT_DIR = Path(__file__).parent.absolute().parent

class_list = [
    'alkane', 'methyl', 'alkene', 'alkyne', 'alcohols', 'amines', 'nitriles',
    'aromatics', 'alkyl halides', 'esters', 'ketones', 'aldehydes', 
    'carboxylic acids', 'ether', 'acyl halides', 'amides', 'nitro'
]


def main(config_path, generate_results):
    # Load configuration file
    with open(config_path, 'r') as config_file:
        config = json.load(config_file)

    # Assign configuration variables
    exp_name = config['exp_name']
    split_at = config['split_at']
    total_epochs = config['total_epochs']
    lr = config['lr']
    gamma = config['gamma']
    seed = config['seed']
    batch_size = config['batch_size']
    num_classes = config['num_classes']
    log_interval = config['log_interval']
    splits = config['splits']

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    model_path = os.path.join(ROOT_DIR, 'models') 
    Path(os.path.join(model_path, exp_name)).mkdir(parents=True, exist_ok=True)
    model_path = os.path.join(model_path, exp_name)

    results_path = os.path.join(ROOT_DIR, 'results')
    Path(os.path.join(results_path, exp_name)).mkdir(parents=True, exist_ok=True)
    results_path = os.path.join(results_path, exp_name) 

    main_df = os.environ['DF']
    dataset = get_dataset(main_df, False)
    dataset = dataset.reset_index(drop=True)
    dataset_slice = dataset[['cano_smi','concat_label', 'spectrum']]

    kf = KFold(n_splits=splits, shuffle=True)

    # Run only one fold
    train_index, test_index = next(kf.split(dataset_slice))
    
    train_df, test_df = dataset_slice.loc[train_index], dataset_slice.loc[test_index]

    test_df.to_pickle(results_path + '/testdf1.pkl')

    train_dataset = IrDataset(
        df=train_df,
        num_classes=num_classes
    )

    test_dataset = IrDataset(
        df=test_df,
        num_classes=num_classes
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=0)

    model, optimizer, criterion, scheduler, device = model_init(lr, num_classes, split_at)
    print('Device: {}'.format(device))

    split_train_acc = []
    split_train_loss = []
    split_test_acc = []
    split_test_loss = []
    plot_logs = []

    start_time = time.time()

    for epoch in range(total_epochs):
        train_loss, tr_acc1, tr_acc0, model = train(model, optimizer, scheduler, criterion, train_loader, device, split_at)
        test_loss, ts_acc1, ts_acc0, preds, raw_preds = test(model, epoch, total_epochs, criterion, test_loader, device, split_at)

        if epoch % log_interval == 0:
            print('Epoch {} : tr_acc :{} {}, tr_loss :{}, ts_acc :{} {}, ts_loss :{}'.format(
                epoch, tr_acc1, tr_acc0, train_loss, ts_acc1, ts_acc0, test_loss))

        scheduler.step()

        split_train_acc.append((tr_acc1, tr_acc0))
        split_train_loss.append(train_loss)
        split_test_acc.append((ts_acc1, ts_acc0))
        split_test_loss.append(test_loss)
        plot_logs.append([epoch, tr_acc1, tr_acc0, train_loss, ts_acc1, ts_acc0, test_loss])

    torch.save(model.state_dict(), model_path + '/{}_1.pth'.format(exp_name))

    with open(results_path + '/log_file_{}.txt'.format(exp_name), 'a+') as file:
        file.write('{} : tr_acc :{} {}, tr_loss :{}, ts_acc :{} {}, ts_loss :{}'.format(
            exp_name, tr_acc1, tr_acc0, train_loss, ts_acc1, ts_acc0, test_loss
        ))
        file.write('\n')

    np.save(results_path + '/1_tr_acc', split_train_acc)
    np.save(results_path + '/1_tr_loss', split_train_loss)
    np.save(results_path + '/1_ts_acc', split_test_acc)
    np.save(results_path + '/1_ts_loss', split_test_loss)
    np.save(results_path + '/1_plot_logs', plot_logs)

    np.save(results_path + '/1_preds', preds)
    np.save(results_path + '/1_rawpreds', raw_preds)

    print("--- %s seconds ---" % (time.time() - start_time))

    if generate_results:
        
        print(f"\nGenerating report for {exp_name}:")
        f1_dict, _ = generate_report(results_path, [1])
        plot_f1_scores([f1_dict], [results_path], class_list)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and test model on one fold")
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration file")
    parser.add_argument("--results", action="store_true", help="Flag to generate results")

    args = parser.parse_args()
    
    main(args.config, args.results)

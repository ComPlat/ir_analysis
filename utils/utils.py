from sklearn.metrics import f1_score
import numpy as np
import pandas as pd

from rdkit import Chem

def convert_to_binary(predictions, threshold):
    main_list = []
    for i in predictions:
        results = (i>threshold).float()*1
        main_list.append(np.array(results))
    return main_list


def pred_acc(prediction, label):

    label = np.asarray(label).reshape(-1)

    total_1 = np.argwhere(label==1.0).shape[0]
    total_0 = np.argwhere(label==0.0).shape[0]

    label_where_1 = [x[0] for x in np.argwhere(label==1.0)]
    label_where_0 = [x[0] for x in np.argwhere(label==0.0)]
    pred_where_1 = [x[0] for x in np.argwhere(prediction==1.0)]
    pred_where_0 = [x[0] for x in np.argwhere(prediction==0.0)]

    correct_0 = [x for x in pred_where_0 if x in label_where_0]
    correct_1 = [x for x in pred_where_1 if x in label_where_1]
    
    if len(correct_1) > 0 :
        acc_1 = ( len(correct_1) / total_1 ) * 100
    else :
        acc_1 = 0.0

    if len(correct_0) > 0 :
        acc_0 = ( len(correct_0) / total_0 ) * 100
    else : 
        acc_0 = 0.0

    return acc_1 , acc_0

def molecule_f1(pred_arr, actual_arr):
    score = []
    for i in range(pred_arr.shape[1]):
        score.append(f1_score(pred_arr[:,i], actual_arr[:,i]))

    return score


def group_accuracy(pred_arr, actual_arr,agg=None):
    output = []
    for i in range(len(pred_arr)):
        _acc_1 , _acc_0 = pred_acc(pred_arr[i], actual_arr[i])
        output.append((_acc_1, _acc_0))
    if agg=='mean':
        acc_1, acc_0 = np.asarray([x[0] for x in output]).mean(), np.asarray([x[1] for x in output]).mean()
    if agg=='sum' :
        acc_1, acc_0 = np.asarray([x[0] for x in output]).sum(), np.asarray([x[1] for x in output]).sum()
    
    return acc_1, acc_0

def group_f1(pred_arr, actual_arr):
    output = []
    for i in range(len(pred_arr)):
        output.append(f1_score( actual_arr[i], pred_arr[i], average='weighted'))
    mean_output = np.mean(output)
    return mean_output






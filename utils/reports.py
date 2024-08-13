import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

ROOT_DIR = Path(__file__).parent.absolute().parent


from sklearn.metrics import balanced_accuracy_score, accuracy_score, f1_score, roc_curve, auc, confusion_matrix

# Define the list of classes
class_list = [
    'alkane', 'methyl', 'alkene', 'alkyne', 'alcohols', 'amines', 'nitriles',
    'aromatics', 'alkyl halides', 'esters', 'ketones', 'aldehydes', 
    'carboxylic acids', 'ether', 'acyl halides', 'amides', 'nitro'
]

def misclass(y_true, y_pred):
    """
    Calculate misclassification statistics and plot them.
    """
    score_dict = {}
    for i in range(y_true.shape[0]):
        total_groups = y_true[i].sum()
        correct_groups = np.sum((y_true[i] + y_pred[i]) == 2)
        score_dict[i] = {
            'total_groups': total_groups,
            'correct': correct_groups,
            'incorrect': total_groups - correct_groups
        }

    scoredf = pd.DataFrame.from_dict(score_dict, orient='index')
    grouped_df = scoredf.groupby('total_groups').mean()

    grouped_df.plot(kind='bar', stacked=False)
    plt.xlabel('Total functional groups per molecule')
    plt.ylabel('Test set mean')
    plt.legend(title='Predictions', bbox_to_anchor=(0.3, 1), loc='upper right')
    plt.grid(True, which='both', alpha=0.3)
    plt.show()

def load_prediction_files(folder, fold_indices):
    """
    Load the prediction files for the specified folder and folds.
    """
    pred_files = []
    for i in fold_indices:
        file_path = os.path.join(folder, f'{i}_preds.npy')
        pred_files.append(np.load(file_path, allow_pickle=True))
    return pred_files

def calculate_metrics_per_class(pred_files, num_classes):
    """
    Calculate accuracy, balanced accuracy, F1 score, and confusion matrices for each class.
    """
    y_trues = [np.vstack(pred[:, 1]) for pred in pred_files]
    y_preds = [np.vstack(pred[:, 0]) for pred in pred_files]

    score_dict = {}
    bal_dict = {}
    f1_dict = {}
    conf_dict = {}

    for i in range(num_classes):
        true_fgs = [y_true[:, i] for y_true in y_trues]
        pred_fgs = [y_pred[:, i] for y_pred in y_preds]

        accuracies = [accuracy_score(true_fg, pred_fg) for true_fg, pred_fg in zip(true_fgs, pred_fgs)]
        balanced_accuracies = [balanced_accuracy_score(true_fg, pred_fg) for true_fg, pred_fg in zip(true_fgs, pred_fgs)]
        f1_scores = [f1_score(true_fg, pred_fg) for true_fg, pred_fg in zip(true_fgs, pred_fgs)]

        score_dict[i] = np.mean(accuracies)
        bal_dict[i] = np.mean(balanced_accuracies)
        f1_dict[i] = [np.mean(f1_scores), np.std(f1_scores)]
        conf_dict[i] = [np.concatenate(true_fgs), np.concatenate(pred_fgs)]
    
    return score_dict, bal_dict, f1_dict, conf_dict

def generate_report(folder, fold_indices):
    """
    Generate and print the report for the specified folder.
    """
    pred_files = load_prediction_files(folder, fold_indices)
    acc_dict, bal_dict, f1_dict, conf_dict = calculate_metrics_per_class(pred_files, len(class_list))

    print('Functional Groups with F1 Score below 0.8:')
    for k, v in f1_dict.items():
        if v[0] < 0.80:
            print(f'{class_list[k]}: {v}')

    print('\nPer-Class Metrics:')
    for idx, fg in enumerate(class_list):
        print(f'FG: {fg} | Balanced Accuracy: {100 * bal_dict[idx]:.2f}% | F1: {f1_dict[idx][0]:.2f} (STD: {f1_dict[idx][1]:.2f})')

    return f1_dict, conf_dict

def plot_f1_scores(f1_dicts, folder_list, class_list, **kwargs):
    """
    Plot F1 scores for multiple folders with class names on the x-axis.
    """
    plt.figure(figsize=(20, 10))
    
    exp_name = kwargs.get('exp_name', 'default_experiment_name')

    for f1_dict, label in zip(f1_dicts, folder_list):
        plt.plot(class_list, [x[0] for x in f1_dict.values()], marker='o', label=label)

    plt.xlabel('Functional Groups')
    plt.ylabel('F1 Score')
    plt.xticks(rotation=45, ha="right")  # Rotate x-axis labels for better readability
    plt.grid(True, which='both', alpha=0.3)
    plt.legend(loc='upper left')

    # Save plot
    plots_path = os.path.join(ROOT_DIR, 'plots')
    Path(plots_path).mkdir(parents=True, exist_ok=True)
    plot_path = os.path.join(plots_path, f'{exp_name}_f1_scores.png')
    plt.savefig(plot_path)
    plt.close()
    print(f'F1 score plot saved to {plot_path}')


if __name__ == "__main__":

    # List of folders to analyze
    folder_list = ['nist_1200']
    
    f1_dicts = []
    for folder in folder_list:
        print(f"\nGenerating report for {folder}:")
        f1_dict, _ = generate_report(folder)
        f1_dicts.append(f1_dict)

    plot_f1_scores(f1_dicts, folder_list, class_list)

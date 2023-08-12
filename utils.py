import numpy as np
import pandas as pd
import argparse
import random
import os
import torch

def Config():
    parser = argparse.ArgumentParser()

    parser.add_argument('--epochs', '-e', type=int, default=30)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--batch_size', '-b', type=int, default=32)
    parser.add_argument('--transform', '-tf', default = 'low')

    config = parser.parse_args()
    return config

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def analysis(result):
    """
    got csv(df) to analysis several metrics

    accuracy, recall, f1-score of each class
    """
    accuracy = (result['infer'] == result['label']).mean()
    metrics = pd.crosstab(result['infer'], result['label'])
    recall_metrics = pd.crosstab(result['infer'], result['label'], normalize='columns')
    precision_metrics = pd.crosstab(result['infer'], result['label'], normalize='index')

    cancer_index = (result['label'] != 0)
    recall_cancer = (result['infer'][cancer_index] != 0).mean()
    precision_cancer = (result['label'][result['infer'] != 0] != 0).mean()
    f1_cancer = (recall_cancer * precision_cancer) / (recall_cancer + precision_cancer)

    def harmonics(grade):
        return (recall_metrics[grade][grade] * precision_metrics[grade][grade]) / (
                recall_metrics[grade][grade] + precision_metrics[grade][grade])

    print('-' * 30)
    print(f'Accuracy : {accuracy:.3f}')
    print('\n', end='')
    print(f'Recall for Benign : {recall_metrics[0][0]:.3f}')
    print(f'Precision for Benign : {precision_metrics[0][0]:.3f}')
    print(f'f1-score for Benign : {harmonics(0):.3f}')
    print('\n', end='')
    print(f'Recall for WD : {recall_metrics[1][1]:.3f}')
    print(f'Precision for WD : {precision_metrics[1][1]:.3f}')
    print(f'f1-score for WD : {harmonics(1):.3f}')
    print('\n', end='')
    print(f'Recall for MD : {recall_metrics[2][2]:.3f}')
    print(f'Precision for MD : {precision_metrics[2][2]:.3f}')
    print(f'f1-score for MD : {harmonics(2):.3f}')
    print('\n', end='')
    print(f'Recall for PD : {recall_metrics[3][3]:.3f}')
    print(f'Precision for PD : {precision_metrics[3][3]:.3f}')
    print(f'f1-score for PD : {harmonics(3):.3f}')
    print('\n', end='')
    print(f'Recall for Cancer : {recall_cancer:.3f}')
    print(f'Precision for Cancer : {precision_cancer:.3f}')
    print(f'f1-score for Cancer : {f1_cancer:.3f}')
    print('\n', end='')

    print('-' * 30)
    print('Confusion Matrix : ')
    print(metrics)

    return

def multi_task_analysis(result):
    analysis(result)

    print('-' * 30)
    accuracy = (result['infer_organ'] == result['organ']).mean()
    metrics = pd.crosstab(result['infer_organ'], result['organ'])
    recall_metrics = pd.crosstab(result['infer_organ'], result['organ'], normalize='columns')
    precision_metrics = pd.crosstab(result['infer_organ'], result['organ'], normalize='index')

    def harmonics(grade):
        return (recall_metrics[grade][grade] * precision_metrics[grade][grade]) / (
                recall_metrics[grade][grade] + precision_metrics[grade][grade])

    print('-' * 30)
    print(f'Accuracy : {accuracy:.3f}')
    print('\n', end='')
    print(f'Recall for colon : {recall_metrics[0][0]:.3f}')
    print(f'Precision for colon : {precision_metrics[0][0]:.3f}')
    print(f'f1-score for colon : {harmonics(0):.3f}')
    print('\n', end='')
    print(f'Recall for prostate : {recall_metrics[1][1]:.3f}')
    print(f'Precision for prostate : {precision_metrics[1][1]:.3f}')
    print(f'f1-score for prostate : {harmonics(1):.3f}')
    print('\n', end='')
    print(f'Recall for gastric : {recall_metrics[2][2]:.3f}')
    print(f'Precision for gastric : {precision_metrics[2][2]:.3f}')
    print(f'f1-score for gastric : {harmonics(2):.3f}')
    print('\n', end='')

    print('-' * 30)
    print('Confusion Matrix : ')
    print(metrics)
    return

import numpy as np
import pandas as pd
import random
import os
import torch
from tqdm.auto import tqdm
from utils import Config, seed_everything, analysis
from train import train_one_organ
from dataset import colon_test_dataloader
from models import single_organ_model


def colon_train_test(network, config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    network = train_one_organ(network, config, 'colon')

    result = pd.DataFrame(columns=['infer', 'label'])
    network.to(device)
    TestDataloader = colon_test_dataloader()

    print('inferring...')

    for idx, i in enumerate(iter(TestDataloader)):
        img = i[0].to(device)
        label = i[1]
        result.loc[idx] = [torch.argmax(network((img))).item(), label[0].item()]

    base_path = 'C:\\Users\\User\\Desktop\\General-Cancer-Classification\\results\\'

    acc = sum(result.infer == result.label) / len(result)
    result.to_csv(base_path + f'infer_result_acc_{acc:.5f}.csv')

    analysis(result)


if __name__ == "__main__":
    seed_everything(42)
    config = Config()
    network = single_organ_model
    colon_train_test(network, config)

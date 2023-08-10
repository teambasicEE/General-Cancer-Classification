import numpy as np
import pandas as pd
import random
import os
import torch
from tqdm.auto import tqdm
from utils import Config, seed_everything, analysis
from train import train_dann
from dataset import total_test_dataloader
from models import DANN_model



def dann_train_test(network, config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    network = train_dann(network, config, 'colon')

    result = pd.DataFrame(columns=['infer_label', 'label', 'infer_organ', 'organ'])
    network.to(device)
    TestDataloader = total_test_dataloader()

    print('inferring...')

    for idx, i in enumerate(iter(TestDataloader)):
        img = i[0].to(device)
        label = i[1]
        organ = i[2]
        result.loc[idx] = [torch.argmax(network((img))[0]).item(), label[0].item(), torch.argmax(network((img))[1]).item(), organ[0].item()]

    base_path = 'C:\\Users\\User\\Desktop\\General-Cancer-Classification\\results\\'

    acc = sum(result.infer == result.label) / len(result)
    result.to_csv(base_path + f'infer_result_acc_{acc:.5f}.csv')

    analysis(result)


if __name__ == "__main__":
    seed_everything(42)
    config = Config()
    network = single_task_model
    colon_train_test(network, config)

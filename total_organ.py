import pandas as pd
import torch
from utils import Config, seed_everything, analysis
from train import train_single_task
from dataset import total_test_dataloader
from models import single_task_model


def total_train_test(network, config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    network = train_single_task(network, config, 'total')

    result = pd.DataFrame(columns=['infer', 'label', 'organ'])
    network.to(device)
    TestDataloader = total_test_dataloader()

    print('inferring...')

    for idx, i in enumerate(iter(TestDataloader)):
        img = i[0].to(device)
        label = i[1]
        organ = i[2]
        result.loc[idx] = [torch.argmax(network(img)).item(), label.item(), organ.item()]

    base_path = 'C:\\Users\\User\\Desktop\\General-Cancer-Classification\\results\\'

    acc = sum(result.infer == result.label) / len(result)
    result.to_csv(base_path + f'total_infer_result_acc_{acc:.5f}.csv')

    analysis(result)


if __name__ == "__main__":
    seed_everything(42)
    config = Config()
    network = single_task_model
    total_train_test(network, config)

import pandas as pd
import torch
from utils import Config, seed_everything, multi_task_analysis
from train import train_multi_task
from dataset import total_test_dataloader
from models import multi_task_model


def multi_task_train_test(network, config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    network = train_multi_task(network, config, 'mutli_task')

    result = pd.DataFrame(columns=['infer', 'label', 'infer_organ', 'organ'])
    network.to(device)
    TestDataloader = total_test_dataloader()

    print('inferring...')

    for idx, i in enumerate(iter(TestDataloader)):
        img = i[0].to(device)
        label = i[1]
        organ = i[2]
        result.loc[idx] = [torch.argmax(network(img)[0]).item(), label.item(), torch.argmax(network(img)[1]).item(), organ.item()]

    base_path = 'C:\\Users\\User\\Desktop\\General-Cancer-Classification\\results\\'

    acc = sum(result['infer'] == result['label']) / len(result)
    result.to_csv(base_path + f'multi_task_infer_result_acc_{acc:.5f}.csv')

    multi_task_analysis(result)


if __name__ == "__main__":
    seed_everything(42)
    config = Config()
    network = multi_task_model
    multi_task_train_test(network, config)

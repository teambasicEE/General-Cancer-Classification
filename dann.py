import numpy as np
import pandas as pd
import torch
from utils import Config, seed_everything, multi_task_analysis
from train import train_multi_task
from models import DANN_model
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from dataset import total_test_dataloader, colon_train_dataloader, prostate_train_dataloader, gastric_train_dataloader

def dann_train_test(network, config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    network = train_multi_task(network, config, 'dann')

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

    if config.sample:
        sampled = 'True'
    else:
        sampled = 'False'

    acc = sum(result['infer'] == result['label']) / len(result)
    result.to_csv(base_path + f'dann_infer_sampling_{sampled}_result_acc_{acc:.5f}.csv')

    multi_task_analysis(result)

    return network
def tsne(network):
    ColonDataloader = colon_train_dataloader(50)
    ProstateDataloader = prostate_train_dataloader(50)
    GastricDataloader = gastric_train_dataloader(50)

    colon_data= next(iter(ColonDataloader))
    prostate_data = next(iter(ProstateDataloader))
    gastric_data = next(iter(GastricDataloader))

    feature_model = list(network.children())[0]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    feature_model.to(device)
    colon_data = colon_data[0].to(device)
    prostate_data = prostate_data[0].to(device)
    gastric_data = gastric_data[0].to(device)

    colon_vector = feature_model(colon_data)
    prostate_vector = feature_model(prostate_data)
    gastric_vector = feature_model(gastric_data)

    df = pd.DataFrame(np.concatenate([colon_vector.cpu().detach().numpy(), prostate_vector.cpu().detach().numpy(), gastric_vector.cpu().detach().numpy()], 0))
    tsne_np = TSNE(n_components=3).fit_transform(df)
    tsne_df = pd.DataFrame(tsne_np, columns=['colon', 'prostate', 'gastric'])

    tsne_df_0 = tsne_df.loc[:50]
    tsne_df_1 = tsne_df.loc[50:100]
    tsne_df_2 = tsne_df.loc[100:]

    #
    return



if __name__ == "__main__":
    seed_everything(42)
    config = Config()
    network = DANN_model
    dann_train_test(network, config)
    # tsne(network)

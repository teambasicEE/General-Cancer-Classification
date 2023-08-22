import torch
import pandas as pd
from dataset import total_test_dataloader
from utils import analysis

colon_model = torch.load('C:\\Users\\User\\Desktop\\General-Cancer-Classification\\results\\colon_epoch_30_batch_42_0.001_best_acc.pt')
prostate_model = torch.load('C:\\Users\\User\\Desktop\\General-Cancer-Classification\\results\\gastric_epoch_30_batch_40_0.001_best_acc.pt')
gastric_model = torch.load('C:\\Users\\User\\Desktop\\General-Cancer-Classification\\results\\prostate_epoch_30_batch_40_0.001_best_acc.pt')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

result = pd.DataFrame(columns=['infer', 'label'])
colon_model.to(device)
prostate_model.to(device)
gastric_model.to(device)
TestDataloader = total_test_dataloader()

print('inferring...')

for idx, i in enumerate(iter(TestDataloader)):
    img = i[0].to(device)
    label = i[1]
    output = colon_model(img) + prostate_model(img) + gastric_model(img)
    result.loc[idx] = [torch.argmax(output).item(), label.item()]

base_path = 'C:\\Users\\User\\Desktop\\General-Cancer-Classification\\results\\'

acc = sum(result.infer == result.label) / len(result)
result.to_csv(base_path + f'ensemble_infer_result_acc_{acc:.5f}.csv')

analysis(result)

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
import wandb
from dataset import colon_train_dataloader, colon_valid_dataloader, prostate_train_dataloader, prostate_valid_dataloader
# ,gastric_train_dataloader, gastric_valid_dataloader, total_train_dataloader, total_valid_dataloader
from utils import Config, seed_everything


def train_single_task(network, config, organ):
    """
    training function for one-organ datas
    """
    wandb.init(
        project='general-cancer-classification',

        config={
            'lr': config.lr,
            'model': 'EfficientnetB0',
            'dataset': organ,
            'epochs': config.epochs,
            'batch_size': config.batch_size
        }
    )

    if organ == 'colon':
        TrainDataloader = colon_train_dataloader(batch_size = config.batch_size)
        ValidDataloader = colon_valid_dataloader(batch_size = config.batch_size)
    elif organ == 'prostate':
        TrainDataloader = prostate_train_dataloader(batch_size=config.batch_size)
        ValidDataloader = prostate_valid_dataloader(batch_size=config.batch_size)
    elif organ == 'gastric' :
        TrainDataloader = gastric_train_dataloader(batch_size=config.batch_size)
        ValidDataloader = gastric_valid_dataloader(batch_size=config.batch_size)
    else :
        TrainDataloader = total_train_dataloader(batch_size = config.batch_szie)
        ValidDataloader = total_valid_dataloader(batch_size = config.batch_szie)

    criterion = torch.nn.CrossEntropyLoss()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    base_path = 'C:\\Users\\User\\Desktop\\General-Cancer-Classification\\results\\'
    epochs = config.epochs
    lr = config.lr
    batch_size = config.batch_size
    optimizer = torch.optim.Adam(network.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, len(TrainDataloader)//5, 0.0001)

    network.to(device)
    loss = []
    acc = []
    valid_loss = []
    valid_acc = []
    best_acc = 25

    for epoch in tqdm(range(epochs)):
        batch_loss = []
        batch_acc = []

        network.train()
        for img, labels, _ in (iter(TrainDataloader)):
            img, labels = torch.autograd.Variable(img), torch.autograd.Variable(labels)
            img, labels = img.to(device), labels.to(device)

            optimizer.zero_grad()
            output = network(img)
            l = criterion(output, labels)
            l.backward()
            optimizer.step()
            scheduler.step()

            batch_loss.append(l.item())
            batch_acc.append(sum(torch.max(output, dim=1)[1].to(device) == labels) / img.shape[0])
            wandb.log({'train_batch_acc': batch_acc[-1], 'train_batch_loss': batch_loss[-1], 'lr': scheduler.get_last_lr()})

        loss.append(sum(batch_loss[-len(TrainDataloader):]) / len(TrainDataloader))
        acc.append(sum(batch_acc[-len(TrainDataloader):]) / len(TrainDataloader))
        wandb.log({'train_acc' : acc[-1], 'train_loss' : loss[-1]})

        network.eval()
        for img, labels, _ in iter(ValidDataloader):
            img, labels = torch.autograd.Variable(img), torch.autograd.Variable(labels)
            img, labels = img.to(device), labels.to(device)

            with torch.no_grad():
                output = network(img)
                l = criterion(output, labels)

                batch_loss.append(l.item())
                batch_acc.append(sum(torch.max(output, 1)[1].to(device) == labels) / img.shape[0])

        valid_loss.append(sum(batch_loss[-len(ValidDataloader):]) / len(ValidDataloader))
        valid_acc.append(sum(batch_acc[-len(ValidDataloader):]) / len(ValidDataloader))
        wandb.log({'valid_acc' : valid_acc[-1], 'valid_loss' : valid_loss[-1]})

        if best_acc >= valid_acc[-1]:
            best_loss = valid_loss[-1]
            best_acc = valid_acc[-1]
            torch.save(network, base_path + f'{organ}_epoch_{epochs}_batch_{batch_size}_{lr}_best_acc.pt')

        print(
            f'\n epoch : {epoch + 1} -- train_loss : {loss[-1]: .5f}, train_acc : {acc[-1]: .5f} valid_loss = {valid_loss[-1]:.5f}, valid_acc = {valid_acc[-1]: .5f}')

    train_result = pd.DataFrame({'train_loss': loss, 'train_acc': acc, 'valid_loss': valid_loss,
                                     'valid_acc': valid_acc})
    train_result.to_csv(base_path + f'{organ}_process.csv', index=False)

    print('-' * 30)
    print('-' * 30)
    print('Train End')
    torch.save(network, base_path + f'{organ}.pt')

    return network


def train_multi_task():

    return

def train_dann():
    return

if __name__ == "__main__":
    seed_everything(42)
    config = Config()
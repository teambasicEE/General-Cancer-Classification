import pandas as pd
import torch
from tqdm.auto import tqdm
import wandb
from dataset import colon_train_dataloader, colon_valid_dataloader, prostate_train_dataloader, prostate_valid_dataloader, gastric_train_dataloader, gastric_valid_dataloader, total_train_dataloader, total_valid_dataloader


def train_single_task(network, config, organ):
    """
    training function for cancer class only datas
    """
    wandb.init(
        project='general-cancer-classification',

        config={
            'lr': config.lr,
            'model': 'EfficientnetB0',
            'dataset': organ,
            'epochs': config.epochs,
            'batch_size': config.batch_size,
            'task' : 'single'
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
        TrainDataloader = total_train_dataloader(batch_size = config.batch_size)
        ValidDataloader = total_valid_dataloader(batch_size = config.batch_size)

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
            temp_lr = scheduler.get_last_lr()
            wandb.log({'train_batch_acc': batch_acc[-1], 'train_batch_loss': batch_loss[-1], 'lr': temp_lr.item()})

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


def train_multi_task(network, config, mode):
    """
    training function for cancer + organ classification
    """
    wandb.init(
        project='general-cancer-classification',

        config={
            'lr': config.lr,
            'model': 'EfficientnetB0',
            'dataset': 'total',
            'epochs': config.epochs,
            'batch_size': config.batch_size,
            'task' : 'dann' if mode == 'dann' else 'multi'
        }
    )

    TrainDataloader = total_train_dataloader(batch_size = config.batch_size)
    ValidDataloader = total_valid_dataloader(batch_size = config.batch_size)

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
    organ_loss = []

    acc = []
    organ_acc = []

    whole_loss = []

    valid_loss = []
    valid_acc = []

    valid_organ_loss = []
    valid_organ_acc = []

    valid_whole_loss = []

    best_acc = 25

    for epoch in tqdm(range(epochs)):
        batch_loss = []
        batch_acc = []
        batch_organ_loss = []
        batch_organ_acc = []
        batch_total_loss = []

        network.train()
        for img, labels, organ in (iter(TrainDataloader)):
            img, labels, organ = torch.autograd.Variable(img), torch.autograd.Variable(labels),torch.autograd.Variable(organ)
            img, labels, organ = img.to(device), labels.to(device), organ.to(device)

            optimizer.zero_grad()
            output = network(img)
            class_loss = criterion(output[0], labels)
            organ_loss = criterion(output[1], organ)

            total_loss = class_loss + organ_loss
            total_loss.backward()

            optimizer.step()
            scheduler.step()

            batch_loss.append(class_loss.item())
            batch_acc.append(sum(torch.max(output[0], dim=1)[1].to(device) == labels) / img.shape[0])

            batch_organ_loss.append(organ_loss.item())
            batch_organ_acc.append(sum(torch.max(output[1], dim=1)[1].to(device) == organ) / img.shape[0])

            batch_total_loss.append(total_loss.item())

        loss.append(sum(batch_loss[-len(TrainDataloader):]) / len(TrainDataloader))
        acc.append(sum(batch_acc[-len(TrainDataloader):]) / len(TrainDataloader))
        organ_loss.append(sum(batch_organ_loss[-len(TrainDataloader):]) / len(TrainDataloader))
        organ_acc.append(sum(batch_organ_acc[-len(TrainDataloader):]) / len(TrainDataloader))
        whole_loss.append(sum(batch_total_loss[-len(TrainDataloader):]) / len(TrainDataloader))
        wandb.log({'train_cancer_acc' : acc[-1], 'train_cancer_loss' : loss[-1],'train_organ_acc' : organ_acc[-1], 'train_organ_loss' :organ_loss[-1],'train_whole_loss' : whole_loss[-1] })

        network.eval()
        for img, labels, organ in iter(ValidDataloader):
            img, labels, organ = torch.autograd.Variable(img), torch.autograd.Variable(labels), torch.autograd.Variable(organ)
            img, labels, organ = img.to(device), labels.to(device), organ.to(device)

            with torch.no_grad():
                output = network(img)
                class_loss = criterion(output[0], labels)
                organ_loss = criterion(output[1], organ)
                whole_loss = class_loss + organ_loss

                batch_loss.append(class_loss.item())
                batch_acc.append(sum(torch.max(output[0], dim=1)[1].to(device) == labels) / img.shape[0])

                batch_organ_loss.append(organ_loss.item())
                batch_organ_acc.append(sum(torch.max(output[1], dim=1)[1].to(device) == organ) / img.shape[0])

                batch_total_loss.append(total_loss.item())

        valid_loss.append(sum(batch_loss[-len(ValidDataloader):]) / len(ValidDataloader))
        valid_acc.append(sum(batch_acc[-len(ValidDataloader):]) / len(ValidDataloader))
        valid_organ_loss.append(sum(batch_organ_loss[-len(ValidDataloader):]) / len(ValidDataloader))
        valid_organ_acc.append(sum(batch_organ_acc[-len(ValidDataloader):]) / len(ValidDataloader))
        valid_whole_loss.append(sum(batch_total_loss[-len(ValidDataloader):]) / len(ValidDataloader))
        wandb.log({'valid_cancer_acc' : acc[-1], 'valid_cancer_loss' : loss[-1],'valid_organ_acc' : valid_organ_acc[-1], 'valid_organ_loss' :valid_organ_loss[-1],'valid_whole_loss' : valid_whole_loss[-1] })

        if best_acc >= valid_acc[-1]:
            best_acc = valid_acc[-1]
            if mode == 'dann':
                torch.save(network, base_path + f'dann_epoch_{epochs}_batch_{batch_size}_{lr}_best_acc.pt')
            else : torch.save(network, base_path + f'multi_class_epoch_{epochs}_batch_{batch_size}_{lr}_best_acc.pt')

        print(
            f'\n epoch : {epoch + 1} -- train_loss : {loss[-1]: .5f}, train_acc : {acc[-1]: .5f} valid_loss = {valid_loss[-1]:.5f}, valid_acc = {valid_acc[-1]: .5f}')

    train_result = pd.DataFrame({'train_loss': loss, 'train_acc': acc, 'train_organ_loss' : organ_loss, 'train_organ_acc' : organ_acc, 'train_whole_loss' : whole_loss, 'valid_loss' :  valid_loss,
                                     'valid_acc': valid_acc,'valid_organ_loss' : valid_organ_loss, 'valid_organ_acc' : valid_organ_acc, 'valid_whole_loss' : valid_whole_loss})
    if mode == 'dann':
        train_result.to_csv(base_path + f'dann_process.csv', index=False)
    else : train_result.to_csv(base_path + f'multi_task_process.csv', index=False)

    print('-' * 30)
    print('-' * 30)
    print('Train End')
    if mode == 'dann':
        torch.save(network, base_path + f'dann_task.pt')
    else :
        torch.save(network, base_path + f'multi_task.pt')

    return network
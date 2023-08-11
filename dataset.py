# Add dataset and Dataloader for each organ, and for mixtures of data
import torch
import cv2
import pandas as pd
from glob import glob
import albumentations as A
from albumentations.pytorch import ToTensorV2


tr_tf = A.Compose([
    A.Resize(512, 512),
    A.HorizontalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.5, contrast_limit=0.3, p=0.3),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

ts_tf = A.Compose([
    A.Resize(512, 512),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

class CustomImageDataset(torch.utils.data.Dataset):
    def __init__(self, mode, transform, label, dir, organ):
        if mode == 'train':
            self.img_labels = label
            self.img_dir = dir
            self.transform = tr_tf
            self.organs = organ

        elif mode == 'valid':
            self.img_labels = label
            self.img_dir = dir
            self.transform = ts_tf
            self.organs = organ

        else :
            self.img_labels = label
            self.img_dir = dir
            self.transform = ts_tf
            self.organs = organ

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        image = cv2.cvtColor(cv2.imread(self.img_dir.iloc[idx]), cv2.COLOR_BGR2RGB)
        label = self.img_labels.iloc[idx]
        organ = self.organs.iloc[idx]
        transformed = self.transform(image=image)
        image = transformed['image']
        return image, label, organ

def file_to_label(filename):
    return int(filename.split('\\')[-1].split('.')[-2].split('_')[-1]) - 1


"""
colon
"""

def colon_data_read():

    colon_path = 'C:\\Users\\User\\Desktop\\colon\\test_2\\colon_45WSIs_1144_08_step05_05\\'
    colon_data_csv = pd.read_csv('C:\\Users\\User\\Desktop\\colon\\colon_split_dataset.csv')

    colon_train_folder = colon_data_csv[colon_data_csv['mode'] == 'train'].folder
    colon_valid_folder = colon_data_csv[colon_data_csv['mode'] == 'valid'].folder
    colon_test_folder = colon_data_csv[colon_data_csv['mode'] == 'test'].folder

    colon_train_dir = []
    colon_valid_dir = []
    colon_test_dir = []

    for i in colon_train_folder:
        colon_train_dir.extend(glob(colon_path + i + '\\*\\*'))
    for i in colon_valid_folder:
        colon_valid_dir.extend(glob(colon_path + i + '\\*\\*'))
    for i in colon_test_folder:
        colon_test_dir.extend(glob(colon_path + i + '\\*\\*'))

    colon_train_label = [file_to_label(i) for i in colon_train_dir]
    colon_valid_label = [file_to_label(i) for i in colon_valid_dir]
    colon_test_label = [file_to_label(i) for i in colon_test_dir]

    return pd.Series(colon_train_dir), pd.Series(colon_train_label), pd.Series(colon_valid_dir), pd.Series(colon_valid_label), pd.Series(colon_test_dir), pd.Series(colon_test_label
                                                                                                                                                               )
def colon_train_dataloader(batch_size):
    train_dir, train_label, valid_dir, valid_label, test_dir, test_label = colon_data_read()
    organ = pd.Series([0 for i in range(len(train_dir))])
    TrainDataset = CustomImageDataset(mode='train', transform=tr_tf, label = train_label, dir = train_dir, organ = organ)
    TrainDataloader = torch.utils.data.DataLoader(TrainDataset, batch_size=batch_size, shuffle=True)
    return TrainDataloader

def colon_valid_dataloader(batch_size):
    train_dir, train_label, valid_dir, valid_label, test_dir, test_label = colon_data_read()
    organ = pd.Series([0 for i in range(len(valid_dir))])
    ValidDataset = CustomImageDataset(mode='valid', transform=ts_tf, label = valid_label, dir = valid_dir, organ = organ)
    ValidDataloader = torch.utils.data.DataLoader(ValidDataset, batch_size=batch_size, shuffle=False)
    return ValidDataloader
def colon_test_dataloader():
    train_dir, train_label, valid_dir, valid_label, test_dir, test_label = colon_data_read()
    organ = pd.Series([0 for i in range(len(test_dir))])
    TestDataset = CustomImageDataset(mode='test', transform=ts_tf, label = test_label, dir = test_dir, organ = organ)
    TestDataloader = torch.utils.data.DataLoader(TestDataset, shuffle=False)
    return TestDataloader

"""
prostate
"""

def prostate_data_read():

    prostate_path = 'C:\\Users\\User\\Desktop\\prostate_harvard\\'

    prostate_train_path = 'C:\\Users\\User\\Desktop\\prostate_harvard\\patches_train_750_v0\\'
    prostate_valid_path = 'C:\\Users\\User\\Desktop\\prostate_harvard\\patches_validation_750_v0\\'
    prostate_test1_path = 'C:\\Users\\User\\Desktop\\prostate_harvard\\patches_test_750_v0\\patho_1\\'
    prostate_test2_path = 'C:\\Users\\User\\Desktop\\prostate_harvard\\patches_test_750_v0\\patho_2\\'

    prostate_train_dir = glob(prostate_train_path + '\\*\\*')
    prostate_valid_dir = glob(prostate_valid_path +  '\\*\\*')
    prostate_test1_dir = glob(prostate_test1_path +  '\\*\\*')
    prostate_test2_dir = glob(prostate_test2_path + '\\*\\*')
    prostate_test_dir = prostate_test1_dir + prostate_test2_dir

    prostate_train_label = [file_to_label(i) +1 for i in prostate_train_dir]
    prostate_valid_label = [file_to_label(i) +1 for i in prostate_valid_dir]
    prostate_test_label = [file_to_label(i) +1 for i in prostate_test_dir]

    return pd.Series(prostate_train_dir), pd.Series(prostate_train_label), pd.Series(prostate_valid_dir), pd.Series(prostate_valid_label), pd.Series(prostate_test_dir), pd.Series(prostate_test_label
                                                                                                                                                               )
def prostate_train_dataloader(batch_size):
    train_dir, train_label, valid_dir, valid_label, test_dir, test_label = prostate_data_read()
    organ = pd.Series([1 for i in range(len(train_dir))])
    TrainDataset = CustomImageDataset(mode='train', transform=tr_tf, label = train_label, dir = train_dir, organ = organ)
    TrainDataloader = torch.utils.data.DataLoader(TrainDataset, batch_size=batch_size, shuffle=True)
    return TrainDataloader

def prostate_valid_dataloader(batch_size):
    train_dir, train_label, valid_dir, valid_label, test_dir, test_label = prostate_data_read()
    organ = pd.Series([1 for i in range(len(valid_dir))])
    ValidDataset = CustomImageDataset(mode='valid', transform=ts_tf, label = valid_label, dir = valid_dir, organ = organ)
    ValidDataloader = torch.utils.data.DataLoader(ValidDataset, batch_size=batch_size, shuffle=False)
    return ValidDataloader
def prostate_test_dataloader():
    train_dir, train_label, valid_dir, valid_label, test_dir, test_label = prostate_data_read()
    organ = pd.Series([1 for i in range(len(test_dir))])
    TestDataset = CustomImageDataset(mode='test', transform=ts_tf, label = test_label, dir = test_dir, organ = organ)
    TestDataloader = torch.utils.data.DataLoader(TestDataset, shuffle=False)
    return TestDataloader

"""
gastric
"""
def prepare_gastric_data(data_label):
    i = 0
    for k in data_label:
        if data_label[i] == '1':
            data_label[i] = '0'

        elif data_label[i] == '2':
            data_label[i] = '0'

        elif data_label[i] == '3':
            data_label[i] = '1'

        elif data_label[i] == '4':
            data_label[i] = '2'

        elif data_label[i] == '5':
            data_label[i] = '3'

        i = i + 1
        data_label = data_label.astype(float)
    return data_label


def gastric_data_read():
    data_dir = ['C:\\Users\\User\\Desktop\\gastric_train\\',
                'C:\\Users\\User\\Desktop\\gastric_valid\\',
                'C:\\Users\\User\\Desktop\\gastric_test\\'
                ]

    dataset1 = glob(data_dir[0] + '\\*\\*')
    dataset_1 = [i + '\\*' for i in dataset_1]
    dataset1 = []
    for i in dataset_1:
        dataset1.extend(glob(i))

    print(dataset1[0])
    print(len(dataset1))
    dataset_2 = glob(data_dir[1] + '*')
    dataset_2 = [i + '\\*' for i in dataset_2]
    dataset2 = []
    for i in dataset_2:
        dataset2.extend(glob(i))

    dataset_3 = glob(data_dir[1] + '*')
    dataset_3 = [i + '\\*' for i in dataset_3]
    dataset3 = []
    for i in dataset_3:
        dataset3.extend(glob(i))

    data_1_label = pd.Series(map(lambda x: x.split('.')[0].split('_')[-1], dataset1))
    data_2_label = pd.Series(map(lambda x: x.split('.')[0].split('_')[-1], dataset2))
    data_3_label = pd.Series(map(lambda x: x.split('.')[0].split('_')[-1], dataset3))


    data_1_label = prepare_gastric_data(data_1_label)
    data_2_label = prepare_gastric_data(data_2_label)
    data_3_label = prepare_gastric_data(data_3_label)

    index1 = data_1_label.loc[data_1_label < 4].index
    index2 = data_2_label.loc[data_2_label < 4].index
    index3 = data_3_label.loc[data_3_label < 4].index

    dataset1 = pd.Series(dataset1).loc[index1]
    dataset2 = pd.Series(dataset2).loc[index2]
    dataset3 = pd.Series(dataset3).loc[index2]

    data_1_label = data_1_label.loc[index1]
    data_2_label = data_2_label.loc[index2]
    data_3_label = data_3_label.loc[index3]

    dataset1 = dataset1.reset_index()
    dataset2 = dataset2.reset_index()
    dataset3 = dataset3.reset_index()
    data_1_label = data_1_label.reset_index()
    data_2_label = data_2_label.reset_index()
    data_3_label = data_3_label.reset_index()

    return dataset1, data_1_label, dataset2, data_2_label, dataset3, data_3_label

def gastric_train_dataloader(batch_size):
    train_dir, train_label, valid_dir, valid_label, test_dir, test_label = gastric_data_read()
    organ = pd.Series([2 for i in range(len(train_dir))])
    TrainDataset = CustomImageDataset(mode='train', transform=tr_tf, label = train_label, dir = train_dir, organ = organ)
    TrainDataloader = torch.utils.data.DataLoader(TrainDataset, batch_size=batch_size, shuffle=True)
    return TrainDataloader

def gastric_valid_dataloader(batch_size):
    train_dir, train_label, valid_dir, valid_label, test_dir, test_label = gastric_data_read()
    organ = pd.Series([2 for i in range(len(valid_dir))])
    ValidDataset = CustomImageDataset(mode='valid', transform=ts_tf, label = valid_label, dir = valid_dir, organ = organ)
    ValidDataloader = torch.utils.data.DataLoader(ValidDataset, batch_size=batch_size, shuffle=False)
    return ValidDataloader

def gastric_test_dataloader():
    train_dir, train_label, valid_dir, valid_label, test_dir, test_label = gastric_data_read()
    organ = pd.Series([2 for i in range(len(test_dir))])
    TestDataset = CustomImageDataset(mode='test', transform=ts_tf, label = test_label, dir = test_dir, organ = organ)
    TestDataloader = torch.utils.data.DataLoader(TestDataset, shuffle=False)
    return TestDataloader

"""
total
+ need to decide how to use data (problem of data imbalance)
Oversampling, undersampling, sampling with weights, ...
"""

def total_train_dataloader(batch_size):
    colon_train_dir, colon_train_label, colon_valid_dir, colon_valid_label, colon_test_dir, colon_test_label = colon_data_read()
    prostate_train_dir, prostate_train_label, prostate_valid_dir, prostate_valid_label, prostate_test_dir, prostate_test_label = prostate_data_read()
    gastric_train_dir, gastric_train_label, gastric_valid_dir, gastric_valid_label, gastric_test_dir, gastric_test_label = gastric_data_read()

    train_dir = pd.concat([colon_train_dir, prostate_train_dir, gastric_train_dir], ignore_index= True)
    train_organ = pd.concat([pd.Series([0 for i in range(len(colon_train_dir))]), pd.Series([1 for i in range(len(prostate_train_dir))]), pd.Series([2 for i in range(len(gastric_train_dir))])], ignore_index = True)
    train_label = pd.concat([colon_train_label, prostate_train_label, gastric_train_label], ignore_index= True)

    TrainDataset = CustomImageDataset(mode='train', transform=tr_tf, label = train_label, dir = train_dir, organ = train_organ)
    TrainDataloader = torch.utils.data.DataLoader(TrainDataset, batch_size=batch_size, shuffle=True)
    return TrainDataloader

def total_valid_dataloader(batch_size):
    colon_train_dir, colon_train_label, colon_valid_dir, colon_valid_label, colon_test_dir, colon_test_label = colon_data_read()
    prostate_train_dir, prostate_train_label, prostate_valid_dir, prostate_valid_label, prostate_test_dir, prostate_test_label = prostate_data_read()
    gastric_train_dir, gastric_train_label, gastric_valid_dir, gastric_valid_label, gastric_test_dir, gastric_test_label = gastric_data_read()

    valid_dir = pd.concat([colon_valid_dir, prostate_valid_dir, gastric_valid_dir], ignore_index= True)
    valid_organ = pd.concat([pd.Series([0 for i in range(len(colon_valid_dir))]), pd.Series([1 for i in range(len(prostate_valid_dir))]), pd.Series([2 for i in range(len(gastric_valid_dir))])], ignore_index = True)
    valid_label = pd.concat([colon_valid_label, prostate_valid_label, gastric_valid_label], ignore_index=True)

    ValidDataset = CustomImageDataset(mode='valid', transform=ts_tf, label = valid_label, dir = valid_dir, organ = valid_organ)
    ValidDataloader = torch.utils.data.DataLoader(ValidDataset, batch_size=batch_size, shuffle=False)
    return ValidDataloader

def total_test_dataloader():
    colon_train_dir, colon_train_label, colon_valid_dir, colon_valid_label, colon_test_dir, colon_test_label = colon_data_read()
    prostate_train_dir, prostate_train_label, prostate_valid_dir, prostate_valid_label, prostate_test_dir, prostate_test_label = prostate_data_read()
    gastric_train_dir, gastric_train_label, gastric_valid_dir, gastric_valid_label, gastric_test_dir, gastric_test_label = gastric_data_read()

    test_dir = pd.concat([colon_test_dir, prostate_test_dir, gastric_test_dir], ignore_index=True)
    test_organ = pd.concat(
        [pd.Series([0 for i in range(len(colon_test_dir))]), pd.Series([1 for i in range(len(prostate_test_dir))]),
         pd.Series([2 for i in range(len(gastric_test_dir))])], ignore_index=True)
    test_label = pd.concat([colon_test_label, prostate_test_label, gastric_test_label], ignore_index=True)

    TestDataset = CustomImageDataset(mode='test', transform=ts_tf, label = test_label, dir = test_dir, organ = test_organ)
    TestDataloader = torch.utils.data.DataLoader(TestDataset, shuffle=False)
    return TestDataloader


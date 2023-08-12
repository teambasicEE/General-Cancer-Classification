# Add dataset and Dataloader for each organ, and for mixtures of data
import torch
import cv2
import pandas as pd
from glob import glob
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision import transforms
import random
from PIL import ImageFilter

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


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

    def __repr__(self):
        return self.__class__.__name__ + '(sigma={})'.format(self.sigma)


high_tf = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.5, 1.)),
    transforms.RandomApply([
        # StainJitter(method='macenko', sigma1=0.1, sigma2=0.1, augment_background=True, n_thread=1),
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
    ], p=0.8),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class CustomImageDataset(torch.utils.data.Dataset):
    def __init__(self, mode, label, dir, organ):
        if mode == 'train':
            self.img_labels = label
            self.img_dir = dir
            self.organs = organ
            self.transform = tr_tf


        elif mode == 'valid':
            self.img_labels = label
            self.img_dir = dir
            self.transform = ts_tf
            self.organs = organ

        else:
            self.img_labels = label
            self.img_dir = dir
            self.transform = ts_tf
            self.organs = organ
        print('apply low-transform')
    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        image = cv2.cvtColor(cv2.imread(self.img_dir.iloc[idx]), cv2.COLOR_BGR2RGB)
        label = self.img_labels.iloc[idx]
        organ = self.organs.iloc[idx]
        transformed = self.transform(image=image)
        image = transformed['image']
        return image, label, organ


class HighTfImageDataset(torch.utils.data.Dataset):
    def __init__(self, mode, label, dir, organ):
        if mode == 'train':
            self.img_labels = label
            self.img_dir = dir
            self.organs = organ
            self.transform = high_tf


        elif mode == 'valid':
            self.img_labels = label
            self.img_dir = dir
            self.transform = ts_tf
            self.organs = organ

        else:
            self.img_labels = label
            self.img_dir = dir
            self.transform = ts_tf
            self.organs = organ
        print('apply high transform')
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

    return pd.Series(colon_train_dir), pd.Series(colon_train_label), pd.Series(colon_valid_dir), pd.Series(
        colon_valid_label), pd.Series(colon_test_dir), pd.Series(colon_test_label
                                                                 )


def colon_train_dataloader(batch_size, tf):
    train_dir, train_label, valid_dir, valid_label, test_dir, test_label = colon_data_read()
    organ = pd.Series([0 for i in range(len(train_dir))])
    if tf == 'high':
        TrainDataset = HighTfImageDataset(mode='train', label=train_label, dir=train_dir, organ=organ)
        TrainDataloader = torch.utils.data.DataLoader(TrainDataset, batch_size=batch_size, shuffle=True)
    else:
        TrainDataset = CustomImageDataset(mode='train', label=train_label, dir=train_dir, organ=organ)
        TrainDataloader = torch.utils.data.DataLoader(TrainDataset, batch_size=batch_size, shuffle=True)
    return TrainDataloader


def colon_valid_dataloader(batch_size):
    train_dir, train_label, valid_dir, valid_label, test_dir, test_label = colon_data_read()
    organ = pd.Series([0 for i in range(len(valid_dir))])
    ValidDataset = CustomImageDataset(mode='valid', label=valid_label, dir=valid_dir, organ=organ)
    ValidDataloader = torch.utils.data.DataLoader(ValidDataset, batch_size=batch_size, shuffle=False)
    return ValidDataloader


def colon_test_dataloader():
    train_dir, train_label, valid_dir, valid_label, test_dir, test_label = colon_data_read()
    organ = pd.Series([0 for i in range(len(test_dir))])
    TestDataset = CustomImageDataset(mode='test', label=test_label, dir=test_dir, organ=organ)
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
    prostate_valid_dir = glob(prostate_valid_path + '\\*\\*')
    prostate_test1_dir = glob(prostate_test1_path + '\\*\\*')
    prostate_test2_dir = glob(prostate_test2_path + '\\*\\*')
    prostate_test_dir = prostate_test1_dir + prostate_test2_dir

    prostate_train_label = [file_to_label(i) + 1 for i in prostate_train_dir]
    prostate_valid_label = [file_to_label(i) + 1 for i in prostate_valid_dir]
    prostate_test_label = [file_to_label(i) + 1 for i in prostate_test_dir]

    return pd.Series(prostate_train_dir), pd.Series(prostate_train_label), pd.Series(prostate_valid_dir), pd.Series(
        prostate_valid_label), pd.Series(prostate_test_dir), pd.Series(prostate_test_label
                                                                       )


def prostate_train_dataloader(batch_size, tf):
    train_dir, train_label, valid_dir, valid_label, test_dir, test_label = prostate_data_read()
    organ = pd.Series([1 for i in range(len(train_dir))])
    if tf == 'high':
        TrainDataset = HighTfImageDataset(mode='train', label=train_label, dir=train_dir, organ=organ)
        TrainDataloader = torch.utils.data.DataLoader(TrainDataset, batch_size=batch_size, shuffle=True)
    else:
        TrainDataset = CustomImageDataset(mode='train', label=train_label, dir=train_dir, organ=organ)
        TrainDataloader = torch.utils.data.DataLoader(TrainDataset, batch_size=batch_size, shuffle=True)
    return TrainDataloader


def prostate_valid_dataloader(batch_size):
    train_dir, train_label, valid_dir, valid_label, test_dir, test_label = prostate_data_read()
    organ = pd.Series([1 for i in range(len(valid_dir))])
    ValidDataset = CustomImageDataset(mode='valid', label=valid_label, dir=valid_dir, organ=organ)
    ValidDataloader = torch.utils.data.DataLoader(ValidDataset, batch_size=batch_size, shuffle=False)
    return ValidDataloader


def prostate_test_dataloader():
    train_dir, train_label, valid_dir, valid_label, test_dir, test_label = prostate_data_read()
    organ = pd.Series([1 for i in range(len(test_dir))])
    TestDataset = CustomImageDataset(mode='test', label=test_label, dir=test_dir, organ=organ)
    TestDataloader = torch.utils.data.DataLoader(TestDataset, shuffle=False)
    return TestDataloader


"""
gastric
"""


def prepare_gastric_data(data_label):
    if data_label == 1:
        data_label = 0

    elif data_label == 2:
        data_label = 0

    elif data_label == 3:
        data_label = 1
    elif data_label == 4:
        data_label = 2

    elif data_label == 5:
        data_label = 3

    return data_label


def gastric_data_read():
    gastric_path = 'C:\\Users\\User\\Desktop\\gastric_data\\'
    gastric_data_csv = pd.read_csv('C:\\Users\\User\\Desktop\\gastric\\gastric_data_seperate.csv')

    gastric_train_folder = gastric_data_csv[gastric_data_csv['Task'] == 'train'].WSI
    gastric_valid_folder = gastric_data_csv[gastric_data_csv['Task'] == 'valid'].WSI
    gastric_test_folder = gastric_data_csv[gastric_data_csv['Task'] == 'test'].WSI

    gastric_train_dir = []
    gastric_valid_dir = []
    gastric_test_dir = []

    for i in gastric_train_folder:
        gastric_train_dir.extend(glob(gastric_path + i + '\\*'))
    for i in gastric_valid_folder:
        gastric_valid_dir.extend(glob(gastric_path + i + '\\*'))
    for i in gastric_test_folder:
        gastric_test_dir.extend(glob(gastric_path + i + '\\*'))

    gastric_train_label = pd.Series([file_to_label(i) + 1 for i in gastric_train_dir])
    gastric_valid_label = pd.Series([file_to_label(i) + 1 for i in gastric_valid_dir])
    gastric_test_label = pd.Series([file_to_label(i) + 1 for i in gastric_test_dir])

    gastric_train_label = gastric_train_label.apply(prepare_gastric_data).astype(float)
    gastric_valid_label = gastric_valid_label.apply(prepare_gastric_data).astype(float)
    gastric_test_label = gastric_test_label.apply(prepare_gastric_data).astype(float)

    index1 = gastric_train_label.loc[gastric_train_label < 4].index
    index2 = gastric_valid_label.loc[gastric_valid_label < 4].index
    index3 = gastric_test_label.loc[gastric_test_label < 4].index

    train_dir = pd.Series(gastric_train_dir).loc[index1]
    valid_dir = pd.Series(gastric_valid_dir).loc[index2]
    test_dir = pd.Series(gastric_test_dir).loc[index3]

    train_label = gastric_train_label.loc[index1]
    valid_label = gastric_valid_label.loc[index2]
    test_label = gastric_test_label.loc[index3]

    return train_dir, train_label, valid_dir, valid_label, test_dir, test_label


def gastric_train_dataloader(batch_size, tf):
    train_dir, train_label, valid_dir, valid_label, test_dir, test_label = gastric_data_read()
    organ = pd.Series([2 for i in range(len(train_dir))])
    if tf == 'high':
        TrainDataset = HighTfImageDataset(mode='train', label=train_label, dir=train_dir, organ=organ)
        TrainDataloader = torch.utils.data.DataLoader(TrainDataset, batch_size=batch_size, shuffle=True)
    else:
        TrainDataset = CustomImageDataset(mode='train', label=train_label, dir=train_dir, organ=organ)
        TrainDataloader = torch.utils.data.DataLoader(TrainDataset, batch_size=batch_size, shuffle=True)
    return TrainDataloader


def gastric_valid_dataloader(batch_size):
    train_dir, train_label, valid_dir, valid_label, test_dir, test_label = gastric_data_read()
    organ = pd.Series([2 for i in range(len(valid_dir))])
    ValidDataset = CustomImageDataset(mode='valid', label=valid_label, dir=valid_dir, organ=organ)
    ValidDataloader = torch.utils.data.DataLoader(ValidDataset, batch_size=batch_size, shuffle=False)
    return ValidDataloader


def gastric_test_dataloader():
    train_dir, train_label, valid_dir, valid_label, test_dir, test_label = gastric_data_read()
    organ = pd.Series([2 for i in range(len(test_dir))])
    TestDataset = CustomImageDataset(mode='test', label=test_label, dir=test_dir, organ=organ)
    TestDataloader = torch.utils.data.DataLoader(TestDataset, shuffle=False)
    return TestDataloader


"""
total
+ need to decide how to use data (problem of data imbalance)
Oversampling, undersampling, sampling with weights, ...
"""


def total_train_dataloader(batch_size, tf):
    colon_train_dir, colon_train_label, colon_valid_dir, colon_valid_label, colon_test_dir, colon_test_label = colon_data_read()
    prostate_train_dir, prostate_train_label, prostate_valid_dir, prostate_valid_label, prostate_test_dir, prostate_test_label = prostate_data_read()
    gastric_train_dir, gastric_train_label, gastric_valid_dir, gastric_valid_label, gastric_test_dir, gastric_test_label = gastric_data_read()

    train_dir = pd.concat([colon_train_dir, prostate_train_dir, gastric_train_dir], ignore_index=True)
    train_organ = pd.concat(
        [pd.Series([0 for i in range(len(colon_train_dir))]), pd.Series([1 for i in range(len(prostate_train_dir))]),
         pd.Series([2 for i in range(len(gastric_train_dir))])], ignore_index=True)
    train_label = pd.concat([colon_train_label, prostate_train_label, gastric_train_label], ignore_index=True)

    if tf == 'high':
        TrainDataset = HighTfImageDataset(mode='train', label=train_label, dir=train_dir, organ=train_organ)
        TrainDataloader = torch.utils.data.DataLoader(TrainDataset, batch_size=batch_size, shuffle=True)
    else:
        TrainDataset = CustomImageDataset(mode='train', label=train_label, dir=train_dir, organ=train_organ)
        TrainDataloader = torch.utils.data.DataLoader(TrainDataset, batch_size=batch_size, shuffle=True)

    return TrainDataloader


def total_valid_dataloader(batch_size):
    colon_train_dir, colon_train_label, colon_valid_dir, colon_valid_label, colon_test_dir, colon_test_label = colon_data_read()
    prostate_train_dir, prostate_train_label, prostate_valid_dir, prostate_valid_label, prostate_test_dir, prostate_test_label = prostate_data_read()
    gastric_train_dir, gastric_train_label, gastric_valid_dir, gastric_valid_label, gastric_test_dir, gastric_test_label = gastric_data_read()

    valid_dir = pd.concat([colon_valid_dir, prostate_valid_dir, gastric_valid_dir], ignore_index=True)
    valid_organ = pd.concat(
        [pd.Series([0 for i in range(len(colon_valid_dir))]), pd.Series([1 for i in range(len(prostate_valid_dir))]),
         pd.Series([2 for i in range(len(gastric_valid_dir))])], ignore_index=True)
    valid_label = pd.concat([colon_valid_label, prostate_valid_label, gastric_valid_label], ignore_index=True)

    ValidDataset = CustomImageDataset(mode='valid', label=valid_label, dir=valid_dir, organ=valid_organ)
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

    TestDataset = CustomImageDataset(mode='test', label=test_label, dir=test_dir, organ=test_organ)
    TestDataloader = torch.utils.data.DataLoader(TestDataset, shuffle=False)
    return TestDataloader

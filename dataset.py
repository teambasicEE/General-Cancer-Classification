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
    def __init__(self, mode, transform, label, dir):
        if mode == 'train':
            self.img_labels = label
            self.img_dir = dir
            self.transform = tr_tf

        elif mode == 'valid':
            self.img_labels = label
            self.img_dir = dir
            self.transform = ts_tf

        else :
            self.img_labels = label
            self.img_dir = dir
            self.transform = ts_tf

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        image = cv2.cvtColor(cv2.imread(self.img_dir.iloc[idx]), cv2.COLOR_BGR2RGB)
        label = [self.img_labels.iloc[idx], 0]              # second 0 for 'colon' among colon, prostate, gastric
        transformed = self.transform(image=image)
        image = transformed['image']
        return image, label

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

    for i in train_folder:
        colon_train_dir.extend(glob(colon_path + i + '\\*\\*'))
    for i in valid_folder:
        colon_valid_dir.extend(glob(colon_path + i + '\\*\\*'))
    for i in test_folder:
        colon_test_dir.extend(glob(colon_path + i + '\\*\\*'))

    colon_train_label = [file_to_label(i) for i in colon_train_dir]
    colon_valid_label = [file_to_label(i) for i in colon_valid_dir]
    colon_test_label = [file_to_label(i) for i in colon_test_dir]

    return pd.Series(colon_train_dir), pd.Series(colon_train_label), pd.Series(colon_valid_dir), pd.Series(colon_valid_label), pd.Series(colon_test_dir), pd.Series(colon_test_label
                                                                                                                                                               )
def colon_train_dataloader(batch_size):
    train_dir, train_label, valid_dir, valid_label, test_dir, test_label = colon_data_read()
    TrainDataset = CustomImageDataset(mode='train', transform=tr_tf, label = train_label, dir = train_dir)
    TrainDataloader = torch.utils.data.DataLoader(TrainDataset, batch_size=batch_size, shuffle=True)
    return TrainDataset, TrainDataloader

def colon_valid_dataloader(batch_size):
    train_dir, train_label, valid_dir, valid_label, test_dir, test_label = colon_data_read()
    ValidDataset = CustomImageDataset(mode='valid', transform=ts_tf, label = valid_label, dir = valid_dir)
    ValidDataloader = torch.utils.data.DataLoader(ValidDataset, batch_size=batch_size, shuffle=False)
    return ValidDataset, ValidDataloader
def colon_test_dataloader():
    train_dir, train_label, valid_dir, valid_label, test_dir, test_label = colon_data_read()
    TestDataset = CustomImageDataset(mode='test', transform=ts_tf, label = test_label, dir = test_dir)
    TestDataloader = torch.utils.data.DataLoader(TestDataset, shuffle=False)
    return TestDataset, TestDataloader

"""
prostate
"""






"""
gastric
"""



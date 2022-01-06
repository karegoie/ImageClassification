import sys
sys.path.append('..')
# from ImageProcessing import center_crop
from ImageClassification.utils.ImageProcessing import ImageTransform

import torch
import numpy as np
import torch.utils.data as data
from glob import glob
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def file_list(path, args):
    dir_list = glob(f'{path}/*')
    if './data/file_list.csv' in dir_list: dir_list.remove('./data/file_list.csv')
    if './data/trash' in dir_list: dir_list.remove('./data/trash')
    real_dir_list = []
    for dir in dir_list:
        img_list = glob(f'{dir}/i*.jpg')
        real_dir_list += img_list

    print('------image loading and throwing away------')

    real_list = []
    for path in tqdm(real_dir_list):
        im = plt.imread(path)
        if np.shape(im)[0] >= args.img_size and np.shape(im)[1] >= args.img_size and np.shape(im)[2] == 3:
            # print(np.shape(im))
            real_list.append(path)

    return real_list


def regular(path_list, df, args):
    freq = df.groupby(['Cluster_a']).count()
    number = min(freq['Name'])

    cluster = {f'{i}': df[df['Cluster_a']==i]['Name'].tolist()[0:number+1] for i in range(args.classes)}

    for i in range(args.classes): cluster[f'{i}'] = cluster[f'{i}'][0:number]

    # print(cluster)
    cluster_list = list(np.array(list(cluster.values())).reshape(1, args.classes*number).squeeze(0))

    real_list = []
    for path in path_list:
        for name in cluster_list:
            if path[7:43] == name: real_list.append(path)

    return real_list

class DoughDataset(data.Dataset):
    def __init__(self, path_list, df, transform, args):
        self.path_list = path_list
        self.annotation = df  # pd.read_csv(f'{args.path}/file_list.csv', encoding='utf-8')
        # print(self.annotation)
        self.transform = transform
        self.args = args

    def __len__(self):
        return len(self.path_list)

    def __getitem__(self, idx):
        img_path = self.path_list[idx]
        img = plt.imread(img_path)
        img = self.transform(img)
        # change if csv changes
        # print(self.annotation)
        # print(self.annotation[self.annotation['Name']==img_path[7:43]])
        index = self.annotation.index[self.annotation['Name']==img_path[7:43]].tolist()[0] # label = self.annotation[self.annotation['hash']==img_path[8:44]].iloc[0, 2]
        # print(index)
        hash = self.annotation[self.annotation['Name']==img_path[7:43]].iloc[0, 0]
        label = self.annotation.iloc[index, 1]
        hot_label = [0] * self.args.classes
        hot_label[label] = 1

        return hash, torch.tensor(img, dtype=torch.float32), torch.tensor(hot_label, dtype=torch.float32)

class PassTheData():
    def __init__(self, args):

        base_path = args.path
        BATCH = args.batch_size
        test_size = args.test_size
        df = pd.read_csv(f'{args.path}/file_list.csv', encoding='utf=8')
        path_list = file_list(base_path, args)
        path_list= regular(path_list=path_list, df=df, args=args)

        # train_test split

        train_img_path_list, test_img_path_list = train_test_split(
            path_list,
            test_size=test_size,
            shuffle=True,
        )

        '''
        img_path_list = file_list(base_path)
        train_img_path_list = img_path_list[test_size:]
        print(img_path_list)
        test_img_path_list = img_path_list[:test_size]
        # print(train_img_path_list)
        '''

        train_dataset = DoughDataset(
            path_list=train_img_path_list,
            transform=ImageTransform(args),
            df=df,
            args=args
        )
        test_dataset = DoughDataset(
            path_list=test_img_path_list,
            transform=ImageTransform(args),
            df=df,
            args=args
        )
        predict_dataset = DoughDataset(
            path_list=path_list,
            transform = ImageTransform(args),
            df=df,
            args=args
        )
        self.train_dataloader = data.DataLoader(dataset=train_dataset, batch_size=BATCH, shuffle=True)
        self.test_dataloader = data.DataLoader(dataset=test_dataset, batch_size=BATCH, shuffle=True)
        self.predict_dataloader = data.DataLoader(dataset=predict_dataset, batch_size=BATCH, shuffle=True)


    def pass_train_dataloader(self):
        return self.train_dataloader

    def pass_test_dataloader(self):
        return self.test_dataloader

    def pass_predict_dataloader(self):
        return self.predict_dataloader


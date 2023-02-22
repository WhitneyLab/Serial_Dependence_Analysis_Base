import pandas as pd
import numpy as np
import pdb 
from datetime import datetime, timedelta
from dis import dis
import torch
import cv2
import vgg_loss
from tqdm import tqdm

import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 22})

pd.options.mode.chained_assignment = None  # default='warn'

data = pd.read_csv("prepped_data.csv")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print('Using device:', device)

crit_vgg = vgg_loss.VGGLoss().to(device)

def get_similarity(image1, image2):

    img1 = cv2.imread(image1) / 255.0
    img2 = cv2.imread(image2) / 255.0
    img1 = cv2.resize(img1, (256,256))
    img2 = cv2.resize(img2, (256,256))
    img1 = torch.from_numpy(img1).to(device)
    img2 = torch.from_numpy(img2).to(device)
    img1 = img1.permute(2,0,1).unsqueeze(0).float()
    img2 = img2.permute(2,0,1).unsqueeze(0).float()

    loss = crit_vgg(img1, img2, target_is_features=False)
    return loss.item()

data['origin'] = data['origin'].str.replace('.jpg', '.jpeg')
data['origin_file'] = '../../../Datasets/ISIC-Archive-Downloader/Data/img/Images/' + data['origin']

data['shifted_origin_1back'] = data['origin'].shift(periods = 1)
data['shifted_origin_file_1back'] = '../../../Datasets/ISIC-Archive-Downloader/Data/img/Images/' + data['shifted_origin_1back']

data['shifted_origin_1forward'] = data['origin'].shift(periods = -1)
data['shifted_origin_file_1forward'] = '../../../Datasets/ISIC-Archive-Downloader/Data/img/Images/' + data['shifted_origin_1forward']

data = data.dropna()
# data = data.reset_index()
# data['similarity'] = None
# for index, row in tqdm(data.iterrows()):
#     row['similarity'] = get_similarity(row['origin_file'], row['shifted_origin_file'])
tqdm.pandas()

data['similarity_1back'] = data.progress_apply(lambda row: get_similarity(row['origin_file'], row['shifted_origin_file_1back']), axis = 1)
print("1back similarity computed!")

data['similarity_1forward'] = data.progress_apply(lambda row: get_similarity(row['origin_file'], row['shifted_origin_file_1forward']), axis = 1)
print("1forward similarity computed!")

print(data.head())

data.to_csv("prepped_data_with_similarity.csv")
print("done!")
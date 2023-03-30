import pandas as pd
import numpy as np
import pdb 
from datetime import datetime, timedelta
from dis import dis
import torch
import cv2
import vgg_loss
from tqdm import tqdm
import lpips

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

def get_image_tensor(image):
    img = cv2.imread(image) / 255.0
    img = cv2.resize(img, (256,256))
    img = torch.from_numpy(img)
    img = img.permute(2,0,1).float() * 2 - 1
    return img

data['origin'] = data['origin'].str.replace('.jpg', '.jpeg')
data['origin_file'] = '../../../Datasets/ISIC-Archive-Downloader/Data/img/Images/' + data['origin']

data['shifted_origin_1back'] = data['origin'].shift(periods = 1)
data['shifted_origin_file_1back'] = '../../../Datasets/ISIC-Archive-Downloader/Data/img/Images/' + data['shifted_origin_1back']

data['shifted_origin_1forward'] = data['origin'].shift(periods = -1)
data['shifted_origin_file_1forward'] = '../../../Datasets/ISIC-Archive-Downloader/Data/img/Images/' + data['shifted_origin_1forward']

data = data.dropna()
data = data.reset_index()

# tqdm.pandas()

loss_fn_vgg = lpips.LPIPS(net='alex')

# data['similarity_1back'] = data.progress_apply(lambda row: loss_fn_vgg(get_image_tensor(row['origin_file']), get_image_tensor(row['shifted_origin_file_1back'])), axis = 1)
# data['similarity_1back'] = [loss_fn_vgg(get_image_tensor(x), get_image_tensor(y)) for x, y in tqdm(zip(data['origin_file'], data['shifted_origin_file_1back']))]
data['similarity_1back'] = None
with torch.no_grad():
    for i in tqdm(range(len(data))):
        data.loc[i, "similarity_1back"] = loss_fn_vgg(get_image_tensor(data.loc[i, "origin_file"]), get_image_tensor(data.loc[i, "shifted_origin_file_1back"]))
print("1back similarity computed!")

# data['similarity_1forward'] = data.progress_apply(lambda row: loss_fn_vgg(get_image_tensor(row['origin_file']), get_image_tensor(row['shifted_origin_file_1forward'])), axis = 1)
# data['similarity_1forward'] = [loss_fn_vgg(get_image_tensor(x), get_image_tensor(y)) for x, y in tqdm(zip(data['origin_file'], data['shifted_origin_file_1forward']))]
data['similarity_1forward'] = None
with torch.no_grad():
    for i in tqdm(range(len(data))):
        data.loc[i, "similarity_1forward"] = loss_fn_vgg(get_image_tensor(data.loc[i, "origin_file"]), get_image_tensor(data.loc[i, "shifted_origin_file_1forward"]))
print("1forward similarity computed!")

print(data.head())

data.to_csv("prepped_data_with_lpips_alex_similarity.csv")
print("done!")
from PIL import Image
import numpy as np
import os
import random
import math
import shutil #copyよう
#
import cv2
from copy import copy
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

import glob
import torchvision.transforms as transforms
import torchvision.utils as vutils
def edge_detection(__input):
    kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], np.float32)  # convolution filter
    with torch.no_grad():
        # [out_ch, in_ch, .., ..] : channel wiseに計算
        edge_k = torch.as_tensor(kernel.reshape(1, 1, 3, 3)) #cudaでやる場合デバイスを指定する

        # エッジ検出はグレースケール化してからやる
        color = __input  # color image [1, 3, H, W]
        gray_kernel = np.array([0.299, 0.587, 0.114], np.float32).reshape(1, 3, 1, 1)  # color -> gray kernel
        gray_k = torch.as_tensor(gray_kernel)
        gray = torch.sum(color * gray_k, dim=1, keepdim=True)  # grayscale image [1, 1, H, W]

        # エッジ検出
        edge_image = F.conv2d(gray, edge_k, padding=1)

    return edge_image

def Get_Pathlists(soutai_dir):
    file_list = []

    path_list = []
    #実行ファイルの絶対パスの取得
    path = os.path.abspath(soutai_dir)
    for root, dirs, files in os.walk(path):
        for fileo in files:
            fulpath = root + "\\" + fileo
            path_list = np.append(path_list,fulpath)
            file_list = np.append(file_list,fileo)
            #print(fulpath)
    return([path_list,file_list])



#file_list = np.full((2,1000),"")

#パスを取得してその中からデータセットにすべきファイルをランダムに抽出する
[path_list,file_list] = Get_Pathlists("./food_images/apple_pie") #ここでデータセットを指定する

    #print(file_list)
    #print(file_list)
seed  = random.seed(1297)
file_samples = random.sample(list(file_list),5) #抽出したい数
seed  = random.seed(1297)
path_samples = random.sample(list(path_list),5)
#print (path_samples)
f = 10

seed  = random.seed(1297)
#抽出したデータセットを256*256にカットしたものと元のものを別ディレクトリに保存する
dataset_path = "food_dataset"
raw_path = "food_dataset\\raw_data"
trimed_path = "food_dataset\\trimed_data"
edge_path = "food_dataset\\edge_data"

if not os.path.exists(dataset_path):
    os.mkdir(dataset_path)

if not os.path.exists(dataset_path + '\\raw_data'):
    os.mkdir(dataset_path + '\\raw_data')

if not os.path.exists(dataset_path + '\\trimed_data'):
    os.mkdir(dataset_path + '\\trimed_data')

if not os.path.exists(dataset_path + '\\edge_data'):
    os.mkdir(dataset_path + '\\edge_data')



#トリムする→ではなく短辺にあわせて正方形にする
for i  in range(len(file_samples)):
    #生データセットをコピーして補完する
    src = path_samples[i]
    copy = raw_path + "\\" + file_samples[i]
    shutil.copyfile(src,copy)

    img = Image.open(path_samples[i])
    w,h = img.size
    crop_size = min(w,h)
    px = 0
    py = 0
    if(w > crop_size):
        px = random.randint(0,w-crop_size)
    if(h > crop_size):
        py = random.randint(0,h-crop_size)
    img_crop = img.crop((px,py,px+crop_size,py+crop_size))

    img_crop.save(trimed_path + '\\' +   file_samples[i])

    #ついでにエッジ検出もおこなう
    trans1 = transforms.ToTensor()

    img_tensor = trans1(img_crop.convert("RGB"))
    img_tensor =  edge_detection(img_tensor)
    #cv2.imwrite(edge_path + '\\' + file_samples[i], img)
    vutils.save_image(img_tensor,edge_path + '\\' + file_samples[i])
    #new_image = img_tensor
    #img_tensor = cv2.cvtColor(img_tensor, cv2.COLOR_GRAY2GRAY) #カラーの場合の処理 BGRをRGBに
    #mg_edge   =   Image.fromarray(img_tensor)
    #img_edge.save(edge_path + '\\' + file_samples[i])
    #print("end")
    







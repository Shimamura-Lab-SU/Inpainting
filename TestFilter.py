#エッジフィルタや鮮鋭化フィルタを試すプロぐラム,適当なディレクトリから画像をとってくる
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

def edge_sobel(__input):
    kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)  # convolution filter
    kernel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], np.float32)  # convolution filter
    with torch.no_grad():
        # [out_ch, in_ch, .., ..] : channel wiseに計算
        edge_k_x = torch.as_tensor(kernel_x.reshape(1, 1, 3, 3)) #cudaでやる場合デバイスを指定する
        edge_k_y = torch.as_tensor(kernel_y.reshape(1, 1, 3, 3)) #cudaでやる場合デバイスを指定する

        # エッジ検出はグレースケール化してからやる
        color = __input  # color image [1, 3, H, W]
        gray_kernel = np.array([0.299, 0.587, 0.114], np.float32).reshape(1, 3, 1, 1)  # color -> gray kernel
        gray_k = torch.as_tensor(gray_kernel)
        gray = torch.sum(color * gray_k, dim=1, keepdim=True)  # grayscale image [1, 1, H, W]

        # エッジ検出
        edge_image_x = F.conv2d(gray, edge_k_x, padding=1)
        edge_image = F.conv2d(edge_image_x, edge_k_y, padding=1)

    return edge_image

def unsharpMask(__input):
    kernel_x = np.array([[-0.11, -0.11, -0.11], [-0.11, 1.88, -0.11], [-0.11, -0.11, -0.11]], np.float32)  # convolution filter
    with torch.no_grad():
        # [out_ch, in_ch, .., ..] : channel wiseに計算
        edge_k = torch.as_tensor(kernel_x.reshape(1,1,3, 3)) #cudaでやる場合デバイスを指定する
        # エッジ検出はグレースケール化してからやる
        color = __input.view(1,3,518,-1) # color image [1, 3, H, W]
        #gray_kernel = np.array([1, 1, 1], np.float32).reshape(1, 3, 1, 1)  # color -> gray kernel
        #gray_k = torch.as_tensor(gray_kernel)
        #gray = torch.sum(color * gray_k, dim=1, keepdim=True)  # grayscale image [1, 1, H, W]

        # エッジ検出
        edge_image = F.conv2d(color, edge_k, padding=1)
        

    return edge_image

def sharpen_filter(__input):
    kernel = np.array([[-2, -2, -2], [-2, 32, -2], [-2, -2, -2]], np.float32) / 16.0  # convolution filter
    with torch.no_grad():
        # [out_ch, in_ch, .., ..] : channel wiseに計算
        sharpen_k = torch.as_tensor(kernel.reshape(1, 1, 3, 3))
        shape = __input.shape
        color = __input.reshape(1,3,shape[1],shape[2])  # color image [1, 3, H, W]
        # channel-wise conv(大事)　3x3 convなのでPadding=1を入れる
        multiband = [F.conv2d(color[:, i:i + 1,:,:], sharpen_k, padding=1) for i in range(3)]
        sharpened_image = torch.cat(multiband, dim=1)
        return sharpened_image


[path_list,file_list] = Get_Pathlists("./TestFilter/raw") #ここでデータセットを指定する

    #print(file_list)
    #print(file_list)
seed  = random.seed(1297)
file_samples = random.sample(list(file_list),10) #抽出したい数
seed  = random.seed(1297)
path_samples = random.sample(list(path_list),10)
#print (path_samples)
f = 10

dataset_path = "TestFilter"
raw_path = "TestFilter\\raw_data"
trimed_path = "TestFilter\\trimed_data"
edge_path = "TestFilter\\edge_data"
unsharp_path = "TestFilter\\unsharp_data"

if not os.path.exists(dataset_path):
    os.mkdir(dataset_path)

if not os.path.exists(dataset_path + '\\raw_data'):
    os.mkdir(dataset_path + '\\raw_data')

if not os.path.exists(dataset_path + '\\trimed_data'):
    os.mkdir(dataset_path + '\\trimed_data')

if not os.path.exists(dataset_path + '\\edge_data'):
    os.mkdir(dataset_path + '\\edge_data')


if not os.path.exists(dataset_path + '\\unsharp_data'):
    os.mkdir(dataset_path + '\\unsharp_data')

for i  in range(len(file_samples)):
    #生データセットをコピーして補完する
    #src = path_samples[i]
    #copy = raw_path + "\\" + file_samples[i]
    #shutil.copyfile(src,copy)

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

    img_tensor = trans1(img.convert("RGB"))
    img_tensor_canny =  edge_detection(img_tensor)
    img_tensor_sobel =  edge_sobel(img_tensor)
    #cv2.imwrite(edge_path + '\\' + file_samples[i], img)
    vutils.save_image(img_tensor_canny,edge_path + '\\Canny'  + file_samples[i])
    vutils.save_image(img_tensor_sobel,edge_path + '\\Sobel'  + file_samples[i])
    #協調してみる
    img_tensor_unsharp = sharpen_filter(img_tensor)
    vutils.save_image(img_tensor_unsharp,unsharp_path + '\\UnSharp_'  + file_samples[i])
    #強調してからエッジ
    img_tensor_target = edge_sobel(sharpen_filter(img_tensor))
    vutils.save_image(img_tensor_target,unsharp_path + '\\UnSharp_Edge_'  + file_samples[i])


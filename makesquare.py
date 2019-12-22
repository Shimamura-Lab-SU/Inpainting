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
            print(fulpath)
    return([path_list,file_list])



#file_list = np.full((2,1000),"")

#パスを取得してその中からデータセットにすべきファイルをランダムに抽出する
[path_list,file_list] = Get_Pathlists("./food_images")

    #print(file_list)
    #print(file_list)
seed  = random.seed(1297)
file_samples = random.sample(list(file_list),10000)
seed  = random.seed(1297)
path_samples = random.sample(list(path_list),10000)
print (path_samples)
f = 10

seed  = random.seed(1297)
#抽出したデータセットを256*256にカットしたものと元のものを別ディレクトリに保存する
dataset_path = "food_dataset"
raw_path = "food_dataset\\raw_data"
trimed_path = "food_dataset\\trimed_data"

if not os.path.exists(dataset_path):
    os.mkdir(dataset_path)

if not os.path.exists(dataset_path + '\\raw_data'):
    os.mkdir(dataset_path + '\\raw_data')

if not os.path.exists(dataset_path + '\\trimed_data'):
    os.mkdir(dataset_path + '\\trimed_data')

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

    







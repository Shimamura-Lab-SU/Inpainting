import random
import math
import torch
#Mdの場所を決める
def Set_Md(_seed,image_size = 256,hall_size = 64):
  #from train_all import padding,image_size,hall_size
  padding = math.floor(hall_size)
  #hall_size = math.floor(image_size / 4)
  d = math.floor(hall_size / 2)
  seed = _seed

  x = random.randint(0 + padding + d,image_size - hall_size - padding + d)
  y = random.randint(0 + padding + d,image_size - hall_size - padding + d)
  return x,y


#Mdの場所に応じてマスクを作成する
def Set_Masks(_masksize,_Mdx,_Mdy,_hallsize,_dtype=torch.float32, batchSize=-1):

  #from train_all import opt
  #from train_all import padding,d,d2,white_channel_float,white_channel_boolen
  d = math.floor(_hallsize / 2)
  #ALL0と1のテンソルを作成する
  black_mask = torch.full((batchSize,1,_masksize,_masksize),0,dtype=_dtype)
  white_mask = torch.full((batchSize,1,_masksize,_masksize),1,dtype=_dtype)
  black_mask[:,:,_Mdx-d:_Mdx+d,_Mdy-d:_Mdy+d] = white_mask[:,:,:_hallsize,:_hallsize]

  return black_mask



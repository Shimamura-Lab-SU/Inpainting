import random
import math
import torch
#Mdの場所を決める
def Set_Md(_seed):
  from train_all import padding,image_size,hall_size
  seed = _seed
  x = random.randint(0 + padding,image_size - hall_size - padding)
  y = random.randint(0 + padding,image_size - hall_size - padding)
  return x,y

#Mdの場所に応じてマスクを作成する
def Set_Masks(_masksize,_Mdx,_Mdy,_hallsize,_dtype=torch.float32):

  from train_all import opt
  #from train_all import padding,d,d2,white_channel_float,white_channel_boolen
  d = math.floor(_hallsize / 2)
  #ALL0と1のテンソルを作成する
  black_mask = torch.full((opt.batchSize,1,_masksize,_masksize),0,dtype=_dtype)
  white_mask = torch.full((opt.batchSize,1,_masksize,_masksize),1,dtype=_dtype)
  black_mask[:,:,_Mdx-d:_Mdx+d,_Mdy-d:_Mdy+d] = white_mask[:,:,:_hallsize,:_hallsize]

  return black_mask



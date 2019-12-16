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
def Set_Masks(_mask,_Mdx,_Mdy,_size,_dtype=torch.float32):
  from train_all import padding,d,d2,white_channel_float,white_channel_boolen
  d = math.floor(_size / 2)
  mask = _mask
  if _dtype == torch.float32:
    mask[:,:,_Mdx-d:_Mdx+d,_Mdy-d:_Mdy+d] = white_channel_float[:,:,:_size,:_size]
  elif _dtype == bool:
    mask[:,:,_Mdx-d:_Mdx+d,_Mdy-d:_Mdy+d] = white_channel_boolen[:,:,:_size,:_size]

  return mask



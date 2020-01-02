from __future__ import print_function
import argparse
import os
from math import log10
import math
import cv2
from copy import copy
import csv
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
import torch.nn.functional as F

from   torch.autograd         import Variable
from   torch.utils.data       import DataLoader
from   networks               import define_G, GANLoss, print_network, define_D_Edge,define_D_Global,define_D_Local,define_Concat,edge_detection
from   data                   import get_training_set, get_test_set
import torch.backends.cudnn as cudnn
from   util                   import save_img, load_img
from   function import Set_Md, Set_Masks
from tqdm import tqdm

import torchvision.transforms as transforms

#import tensorboard as tbx # tensorboardXのインポート[ポイント1]
from torch.utils.tensorboard import SummaryWriter

import random
import time
import datetime
transform_list = [transforms.ToTensor(),
                  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

transform = transforms.Compose(transform_list)

#writer = SummaryWriter(log_dir="logs")# SummaryWriterのインスタンス作成[ポイント2]

# Training settings
parser = argparse.ArgumentParser(description='a fork of pytorch pix2pix')
parser.add_argument('--dataset', required=True, help='facades')
parser.add_argument('--batchSize', type=int, default=1, help='training batch size')
parser.add_argument('--testBatchSize', type=int, default=1, help='testing batch size')
parser.add_argument('--nEpochs', type=int, default=200, help='number of epochs to train for')
parser.add_argument('--input_nc', type=int, default=3, help='input image channels')
parser.add_argument('--output_nc', type=int, default=3, help='output image channels')
parser.add_argument('--ngf', type=int, default=64, help='generator filters in first conv layer')
parser.add_argument('--ndf', type=int, default=64, help='discriminator filters in first conv layer')
parser.add_argument('--lr', type=float, default=0.0004, help='Learning Rate. Default=0.002') #1に変更'(0.0004)
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', help='use cuda?')
parser.add_argument('--threads', type=int, default=4, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--lamb', type=int, default=10, help='weight on L1 term in objective')
parser.add_argument('--G_model', type=str, default='checkpoint/testing_modelG_25.pth', help='model file to use')

opt = parser.parse_args()
Output_Each_Epoch = True #エポックの終了時に訓練結果を画像として出力するか否か

#画像サイズまわりはここで整合性をとりながら入力すること
hall_size = 64 # 穴の大きさ(pic)
Local_Window = 128 #ローカルDiscが参照する範囲の大きさ(pic)
image_size = 256 #入力画像全体のサイズ
#mask_channelは1*256*256の白黒マスク
center = 128#奇数の場合は切り捨てる 
d = 32 #生成窓の半分の大きさ
d2 =  64#LocalDiscriminator窓の半分の大きさ
padding = 64 #Mdが窓を生成し得る位置がが端から何ピクセル離れているか 

disc_weight = 0.0004

print(opt)

if opt.cuda and not torch.cuda.is_available():
  raise Exception("No GPU found, please run without --cuda")

cudnn.benchmark = True

torch.manual_seed(opt.seed)
#if opt.cuda:
#  torch.cuda.manual_seed(opt.seed)

print('===> Loading datasets')
root_path            = "dataset/"
train_set            = get_training_set(root_path + opt.dataset)
test_set             = get_test_set(root_path + opt.dataset)
training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True)
testing_data_loader  = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=opt.testBatchSize, shuffle=False)

max_dataset_num = 7500#データセットの数 (8000コ)
max_test_dataset_num = 100#データセットの数 (2000コ)

train_set.image_filenames = train_set.image_filenames[:max_dataset_num]
test_set.image_filenames = test_set.image_filenames[:max_test_dataset_num]

print('===> Building model')

#3つのタスクのそれぞれのエポック

#先ず学習済みのGeneratorを読み込んで入れる
#netG = torch.load(opt.G_model)
disc_input_nc = 4
disc_outpuc_nc = 1024

#100epoch組
#netG = torch.load("checkpoint/netG_model_epoch_100.pth")
#netD_Global = torch.load("checkpoint/netDg_model_epoch_100.pth")
#netD_Local = torch.load("checkpoint/netDl_model_epoch_100.pth")

#401epoch組(Edgeなし)
#netG = torch.load("checkpoint/netG_model_epoch_401.pth")
#netD_Global = torch.load("checkpoint/netDg_model_epoch_401.pth")
#netD_Local = torch.load("checkpoint/netDl_model_epoch_401.pth")

#Edge有350組
#netD_Edge = torch.load("checkpoint/netDe_1.pth")
#netD_Global = torch.load("checkpoint/netDg_1.pth")
#netD_Local = torch.load("checkpoint/netDl_1.pth")


#netG = torch.load("checkpoint/netG_10_1227.pth")
netG = define_G(4, 3, opt.ngf, 'batch', False, [0])
#netD_Global = torch.load("checkpoint/testing_modelDg_4.pth")  
#netD_Global = torch.load("checkpoint/testing_modelDg1223_10.pth")
#netD_Local = torch.load("checkpoint/testing_modelDl1223_10.pth")
netD_Global = define_D_Global(disc_input_nc , disc_outpuc_nc, opt.ndf,  [0])
netD_Local   = define_D_Local(disc_input_nc , disc_outpuc_nc, opt.ndf,  [0])
netD_Edge     = define_D_Edge(2 , disc_outpuc_nc, opt.ndf,  [0]) #1/1 4→2
net_Concat = define_Concat(2048,1,[0])
net_Concat1 = define_Concat(1024,1,[0])

criterionGAN = GANLoss()
criterionL1 = nn.L1Loss()
criterionMSE = nn.MSELoss()
criterionBCE = nn.BCELoss()#Discriminatorに使うため新設
# setup optimizer
optimizerG = optim.Adadelta(netG.parameters(), lr=opt.lr) # 
optimizerD_Global = optim.Adadelta(netD_Global.parameters(), lr=opt.lr) #
optimizerD_Local = optim.Adadelta(netD_Local.parameters(), lr=opt.lr) #
optimizerD_Edge = optim.Adadelta(netD_Edge.parameters(), lr=opt.lr) #

print('---------- Networks initialized -------------')
print_network(netG)
print_network(netD_Global)
print_network(netD_Local)
print_network(netD_Edge)
print('-----------------------------------------------')


if opt.cuda:
  #
  netD_Global = netD_Global.cuda()
  netD_Local  = netD_Local.cuda()
  netD_Edge   = netD_Edge.cuda()
  netG = netG.cuda()#
  net_Concat = net_Concat.cuda()
  net_Concat1 = net_Concat1.cuda()
  #
  criterionGAN = criterionGAN.cuda()
  criterionL1 = criterionL1.cuda()
  criterionMSE = criterionMSE.cuda()
  criterionBCE = criterionBCE.cuda()
  #
  #real_a_image = real_a_image.cuda()

#ここでCPUに置いてあるreal_aを定義
real_a_image = torch.FloatTensor(opt.batchSize, opt.output_nc, 256, 256)


white_channel_float_128 = torch.full((opt.batchSize,1,128,128), True)
##Mc(inputMask)の定義
mask_channel_float = Set_Masks(image_size,center,center,hall_size,batchSize=opt.batchSize) #古い宣言
mask_channel_boolen = Set_Masks(image_size,center,center,hall_size,bool,batchSize=opt.batchSize)
##Md(RandomMask)の定義


#Mdの場所のためのシード固定
seed = random.seed(1297)


start_date = datetime.date.today()
start_time = datetime.datetime.now()

mask_channel_3d_b = torch.cat((mask_channel_boolen,mask_channel_boolen,mask_channel_boolen),1)
#mask_channel_3d_b = mask_channel_3d_b.cuda()
mask_channel_3d_f = torch.cat((mask_channel_float,mask_channel_float,mask_channel_float),1)
#mask_channel_3d_f = mask_channel_3d_f.cuda()


#エポックごとに出力していく
result_list = []

each_loss_plot_flag = True

def train(epoch,mode=0):

  center = math.floor(image_size / 2)
  d = math.floor(Local_Window / 4) #trim(LocalDiscriminator用の窓)
  d2 = math.floor(Local_Window / 2) #L1Loss用の純粋な生成画像と同じサイズの窓用,所謂Mc

  mask_channel_boolen = Set_Masks(image_size,center,center,hall_size,bool,batchSize=opt.batchSize)
  mask_channel_float = Set_Masks(image_size,center,center,hall_size,batchSize=opt.batchSize)

  epoch_start_time = time.time()
  #loss_plot_array = np.zeros(int(opt.batchSize / max_dataset_num),4) # 1エポックごとにロスを書き込む
  loss_plot_array = np.empty((int(max_dataset_num / opt.batchSize),4))

  result_list.append(str(epoch))
  #0..OnlyGenerator
  #1..OnlyDiscriminator
  #2..Both 
  flag_global = True
  flag_local  = True
  flag_edge   = True

  loss_g_avg = 0
  loss_d_avg = 0
  #------------
  loss_dg_r_avg = 0    #real
  loss_dl_r_avg = 0
  loss_de_r_avg = 0
  loss_dg_f_avg = 0  #fakeの個別Loss
  loss_dl_f_avg = 0
  loss_de_f_avg = 0

  #Generatorの学習タスク
  for iteration, batch in tqdm(enumerate(training_data_loader, 1)):


    #####################################################################
    #ランダムマスクの作成
    #####################################################################

    #Mdの位置
    Mdpos_x,Mdpos_y = Set_Md(seed)
    #Mdを↑の位置に当てはめる
    
    random_mask_float_64 = Set_Masks(image_size,Mdpos_x,Mdpos_y,hall_size,torch.float32,batchSize=opt.batchSize)
    random_mask_boolen_64 = Set_Masks(image_size,Mdpos_x,Mdpos_y,hall_size,bool,batchSize=opt.batchSize)
    random_mask_float_128 = Set_Masks(image_size,Mdpos_x,Mdpos_y,Local_Window,torch.float32,batchSize=opt.batchSize)
    random_mask_boolen_128 = Set_Masks(image_size,Mdpos_x,Mdpos_y,Local_Window,bool,batchSize=opt.batchSize)


    #####################################################################
    #イメージテンソルの定義
    #####################################################################
    real_a_image_cpu = batch[1]#batch[1]が原画像そのまんま
    real_a_image.data.resize_(real_a_image_cpu.size()).copy_(real_a_image_cpu)#realAへのデータ流し込み
    #real_a_image = batch[1]
    #####################################################################
    #先ずGeneratorを起動して補完モデルを行う
    #####################################################################

    #####################################################################
    #Global,LocalDiscriminatorを走らせる
    #####################################################################
    real_a_image_4d = torch.cat((real_a_image,random_mask_float_64),1) #ここ固定マスクじゃない?(12/25)

    #real_a_image_4d = torch.cat((real_a_image,mask_channel_float),1) #ここ固定マスクじゃない?(12/25)
    #real_a_image_4d = real_a_image_4d.cuda() 
    #mask_channel_float = mask_channel_float.cuda()#どうしても必要なためcudaに入れる


    #1stDiscriminator
    if mode==1 or mode==2:
      #マスクとrealbの結合

      #12/17optimizerをzero_gradする
      #128*128の窓を作成 (あとで裏に回れるようにする)
      #真画像とマスクの結合..4次元に
      fake_b_image_raw = netG.forwardWithMasking(real_a_image_4d,hall_size,opt.batchSize) # C(x,Mc) #ここをnetGの機能にする
      
      fake_b_image_raw_4d = torch.cat((fake_b_image_raw,mask_channel_float),1) #catはメインループ内で許可
      #fake_b_image_raw_4d = fake_b_image_raw_4d.cuda()


      if flag_local:
        pred_realD_Local  =  netD_Local.forwardWithTrim(real_a_image_4d.detach(),_xpos = Mdpos_x,_ypos = Mdpos_y,trim_size = Local_Window,batchSize = opt.batchSize)
        pred_fakeD_Local = netD_Local.forwardWithTrimCover(fake_b_image_raw_4d.detach(),_xpos = Mdpos_x,_ypos = Mdpos_y,trim_size = Local_Window,_input_real = real_a_image_4d,hole_size = hall_size,batchSize = opt.batchSize) #pred_falke=D(C(x,Mc),Mc)
      if flag_global:
        pred_fakeD_Global = netD_Global.forwardWithCover(fake_b_image_raw_4d,_input_real = real_a_image_4d,hole_size = hall_size) #pred_falke=D(C(x,Mc),Mc)
        pred_realD_Global =  netD_Global.forward(real_a_image_4d.detach())
      if flag_edge:
        pred_realD_Edge  = netD_Edge.forwardWithTrim(real_a_image_4d.detach(),_xpos = Mdpos_x,_ypos = Mdpos_y,trim_size = Local_Window,batchSize = opt.batchSize)
        pred_fakeD_Edge  = netD_Edge.forwardWithTrimCover(fake_b_image_raw_4d.detach(),_xpos = Mdpos_x,_ypos = Mdpos_y,trim_size = Local_Window,_input_real = real_a_image_4d,hole_size = hall_size,batchSize = opt.batchSize) #pred_falke=D(C(x,Mc),Mc)

      #pred_fakeは偽生成画像を入力としたときの尤度テンソル
      #〇〇〇(残りのパターンは省略)
      if (flag_global == True) and (flag_local == True) and (flag_edge == True):
        #Concatを使って繋げる
        pred_realD = net_Concat(pred_realD_Global,pred_realD_Local)
        pred_fakeD = net_Concat(pred_fakeD_Global,pred_fakeD_Local)

      #〇〇
      if (flag_global == True) and (flag_local == True) and (flag_edge == False):
        #Concatを使って繋げる
        pred_realD = net_Concat(pred_realD_Global,pred_realD_Local)
        pred_fakeD = net_Concat(pred_fakeD_Global,pred_fakeD_Local)

      #〇×
      if (flag_global == True) and (flag_local == False):
        pred_realD = pred_realD_Global
        pred_fakeD = pred_fakeD_Global
      #×〇
      if (flag_global == False) and (flag_local == True):
        pred_realD = pred_realD_Global
        pred_fakeD = pred_fakeD_Global
      
      #真偽はGPUに乗っける
      false_label_tensor = Variable(torch.LongTensor())
      false_label_tensor  = torch.zeros(opt.batchSize,1)
      true_label_tensor = Variable(torch.LongTensor())
      true_label_tensor  = torch.ones(opt.batchSize,1)

      loss_d_realD = criterionBCE(pred_realD, true_label_tensor)
      loss_d_fakeD = criterionBCE(pred_fakeD, false_label_tensor) #ニセモノ-ホンモノをニセモノと判断させたいのでfalse

      #プロット用にGlobalとLocalのLossを上とは別に個別に導出する
      if (flag_global == True):
        #perd_realDはここで破壊する
        pred_realD_Global = net_Concat1.forward1(pred_realD_Global)
        pred_fakeD_Global = net_Concat1.forward1(pred_fakeD_Global)
        loss_d_realD_Global = criterionBCE(pred_realD_Global, true_label_tensor)
        loss_d_fakeD_Global = criterionBCE(pred_fakeD_Global, false_label_tensor)
      if (flag_local == True):
        pred_realD_Local = net_Concat1.forward1(pred_realD_Local)
        pred_fakeD_Local = net_Concat1.forward1(pred_fakeD_Local)
        loss_d_realD_Local  = criterionBCE(pred_realD_Local, true_label_tensor)
        loss_d_fakeD_Local  = criterionBCE(pred_fakeD_Local, false_label_tensor)

      if (flag_edge == True):
        pred_fakeD_Edge = net_Concat1.forward1(pred_fakeD_Edge)
        pred_realD_Edge = net_Concat1.forward1(pred_realD_Edge)
        loss_d_realD_Edge = criterionBCE(pred_realD_Edge, true_label_tensor)
        loss_d_fakeD_Edge = criterionBCE(pred_fakeD_Edge, false_label_tensor)

    #2つのロスの足し合わせ
      if mode == 1:
        loss_d = loss_d_realD + loss_d_fakeD 
      if mode == 2:
        if (flag_edge == True):
          loss_d = (loss_d_fakeD + loss_d_realD + loss_d_fakeD_Edge + loss_d_realD_Edge) * disc_weight
        else:
          loss_d = (loss_d_fakeD + loss_d_realD) * disc_weight
    #backward
      loss_d.backward()


    #optimizerの更新
      if flag_global: 
        optimizerD_Global.step()
      if flag_local: 
        optimizerD_Local.step()
      if flag_edge:
        optimizerD_Edge.step()

    #マスクとrealbの結合
    #real_b_image_4d = torch.cat((real_b_image,mask_channel_float),1)
    #####################################################################
    #Generatorの学習を行う    
    ######################################################################   


    if mode==0 or mode==2:
      
      #optimizerG.zero_grad()
      fake_b_image_raw = netG.forwardWithMasking(real_a_image_4d,hall_size,opt.batchSize) # C(x,Mc)
      #fake_b_image = real_a_image.clone()
      #fake_b_image[:,:,center-d:center+d,center-d:center+d] = fake_b_image_raw[:,:,center-d:center+d,center-d:center+d] 
      fake_b_image_raw_4d = torch.cat((fake_b_image_raw,mask_channel_float),1) #catはメインループ内で許可
      #fake_b_image_raw_4d = fake_b_image_raw_4d.cuda()

      #2ndDiscriminator
      if mode == 2:
        if flag_global: 
          pred_fakeD_Global = netD_Global.forwardWithCover(fake_b_image_raw_4d.detach(),_input_real = real_a_image_4d,hole_size = hall_size) #pred_falke=D(C(x,Mc),Mc)
        if flag_local:
          pred_fakeD_Local = netD_Local.forwardWithTrimCover(fake_b_image_raw_4d.detach(),_xpos = Mdpos_x,_ypos = Mdpos_y,trim_size = Local_Window,_input_real = real_a_image_4d,hole_size = hall_size,batchSize = opt.batchSize) #pred_falke=D(C(x,Mc),Mc)
        if flag_edge:
          pred_fakeD_Edge = netD_Edge.forwardWithTrimCover(fake_b_image_raw_4d.detach(),_xpos = Mdpos_x,_ypos = Mdpos_y,trim_size = Local_Window,_input_real = real_a_image_4d,hole_size = hall_size,batchSize = opt.batchSize) #pred_falke=D(C(x,Mc),Mc)
          
        #pred_fakeは偽生成画像を入力としたときの尤度テンソル
        #〇〇〇
        if (flag_global == True) and (flag_local == True) and (flag_edge == True):
          #pred_realD = net_Concat(pred_realD_Global,pred_realD_Local)#エッジはConcatしない
          pred_fakeD = net_Concat(pred_fakeD_Global,pred_fakeD_Local)

        #〇〇
        if (flag_global == True) and (flag_local == True) and (flag_edge == False):
          #Concatを使って繋げる
          #pred_realD = net_Concat(pred_realD_Global,pred_realD_Local)
          pred_fakeD = net_Concat(pred_fakeD_Global,pred_fakeD_Local)

        #〇×
        if (flag_global == True) and (flag_local == False):
          #pred_realD = pred_realD_Global
          pred_fakeD = pred_fakeD_Global
        #×〇
        if (flag_global == False) and (flag_local == True):
          #pred_realD = pred_realD_Global
          pred_fakeD = pred_fakeD_Global

        if(flag_edge == True):
          pred_fakeD_Edge = net_Concat1.forward1(pred_fakeD_Edge)

		  #reconstructError
      #fake_b_image_masked =  #GPU回避不可能？
      #real_a_image_masked =  #1次元で(61440)出てくるので..
      reconstruct_error = criterionMSE(torch.masked_select(fake_b_image_raw, mask_channel_3d_b),torch.masked_select(real_a_image_4d[:,0:3,:,:], mask_channel_3d_b))# 生成画像とオリジナルの差
      #fake_D_predを用いたエラー
      loss_g = reconstruct_error
      #最初のiterなら記録する (4桁)

      if mode == 2:
        loss_d_fakeD = disc_weight * criterionBCE(pred_fakeD, true_label_tensor) 
        loss_g += loss_d_fakeD
        if flag_edge == True:
          loss_d_fakeD_Edge = disc_weight * criterionBCE(pred_fakeD_Edge, true_label_tensor)
          loss_g += loss_d_fakeD_Edge
      
      loss_g.backward()
		  #Optimizerの更新
      optimizerG.step()
      #plotようにreal_aを貼り付けたもの
      fake_b_image = real_a_image.clone()
      fake_b_image[:,:,center-d:center+d,center-d:center+d] = fake_b_image_raw[:,:,center-d:center+d,center-d:center+d] 


      loss_plot_array[iteration-1][0] = epoch
      loss_plot_array[iteration-1][1] = iteration

      if mode == 0 or mode == 2:
        loss_plot_array[iteration-1][2] = loss_g
      if mode == 1:
        loss_plot_array[iteration-1][3] = loss_d


      loss_g_avg += loss_g
    if(mode != 0):
      loss_d_avg += loss_d
      loss_dg_r_avg += loss_d_realD_Global
      loss_dg_f_avg += loss_d_fakeD_Global
      loss_dl_r_avg += loss_d_realD_Local
      loss_dl_f_avg += loss_d_fakeD_Local
      loss_de_r_avg += loss_d_realD_Edge
      loss_de_f_avg += loss_d_fakeD_Edge

    if(mode == 1 or mode == 2):
      loss_d_avg += loss_d
    #最後のiterならログを記録する
    if(iteration ==  (max_dataset_num / opt.batchSize) ):
      loss_g_avg = loss_g_avg / iteration
      if(mode == 1 or mode == 2):
        loss_d_avg = loss_d_avg / iteration
      result_list.append('{:.4g}'.format(loss_g_avg))
      if(mode == 1 or mode == 2):
        result_list.append('{:.4g}'.format(loss_d_avg))
      if(mode == 1 or mode == 2):
        loss_d_avg = 0
        loss_g_avg = 0
        #フラグが有効なら個別のロスを記録する
      if(each_loss_plot_flag and mode != 0):
        loss_dg_r_avg = loss_dg_r_avg / iteration 
        loss_dg_f_avg = loss_dg_f_avg / iteration 
        loss_dl_r_avg = loss_dl_r_avg / iteration
        loss_dl_f_avg = loss_dl_f_avg / iteration
        loss_de_r_avg = loss_de_r_avg / iteration 
        loss_de_f_avg = loss_de_f_avg / iteration 

        result_list.append(('{:.4g}'.format(loss_d_realD_Global)))
        result_list.append(('{:.4g}'.format(loss_d_fakeD_Global)))
        result_list.append(('{:.4g}'.format(loss_d_realD_Local)))
        result_list.append(('{:.4g}'.format(loss_d_fakeD_Local)))
        result_list.append(('{:.4g}'.format(loss_d_realD_Edge)))
        result_list.append(('{:.4g}'.format(loss_d_fakeD_Edge)))
      else:
        for i in range(6):
          result_list.append(0)
      if mode == 0: #mode0の場合
        result_list.append(0)


      loss_dg_r_avg = 0
      loss_dg_f_avg = 0
      loss_dl_r_avg = 0
      loss_dl_f_avg = 0
      loss_de_r_avg = 0
      loss_de_f_avg = 0

    #####################################################################
    #ログの作成、画像の出力
    #####################################################################
    if mode == 0 or mode == 2:
      if(iteration == 1):
        #初回のみRealA
        if(epoch == 1):
          Plot2Image(real_a_image,TrainRealA_dir_,"/fakeA{}".format(epoch))
        Plot2Image(fake_b_image_raw,TrainFakeB_Raw_dir_,"/fakeB_Raw_{}".format(epoch))
        Plot2Image(fake_b_image,TrainFakeB_dir_,"/fakeB_{}".format(epoch))
        if(flag_edge==True):
          Plot2Image(edge_detection( fake_b_image,False),TrainFakeB_Edge_dir_,"/fakeB_Edge_{}".format(epoch))

      #ロスの出力
      with open(Loss_dir_ + '/loss_log.csv', 'a') as f:
        writer = csv.writer(f)
        writer.writerows(loss_plot_array)



#テスト側
def test(epoch,mode=0):
  loss_g_avg = 0
  loss_d_avg = 0

  loss_dg_r_avg = 0    #real
  loss_dl_r_avg = 0
  loss_de_r_avg = 0
  loss_dg_f_avg = 0  #fakeの個別Loss
  loss_dl_f_avg = 0
  loss_de_f_avg = 0


  flag_local = True
  flag_edge = True
  flag_global = True
  #mode = 2 #いったんここで定義

  center = math.floor(image_size / 2)
  d = math.floor(Local_Window / 4) #trim(LocalDiscriminator用の窓)
  d2 = math.floor(Local_Window / 2) #L1Loss用の純粋な生成画像と同じサイズの窓用,所謂Mc
  image_dir = "dataset/{}/test/b/".format(opt.dataset)
      #####################################################################
    #一定の周期でテストを行う 
    #####################################################################
  test_epoch = 1 #毎エポック

  fake_b_image = torch.Tensor(1,3,image_size,image_size)
  if (epoch == 1 or (epoch % test_epoch) == 0):
    iteration = 1
    for image_name in test_set.image_filenames:
      img = load_img(image_dir + image_name)
      img = transform(img)
      input = Variable(img).view(1,-1,256,256)
      #元イメージは壊す前に出力しておく
      Plot2Image(input,TestRealA_dir_,'/'+ str(epoch)+'_' +image_name) 
      mask = mask_channel_float[0].view(1,-1,256,256) #ここが固定マスクなのは別に問題ではない節

      real_a_image_4d = torch.cat((input,mask),1) #ここ固定マスクじゃない?(12/25)

      fake_b_raw = netG.forwardWithMasking(real_a_image_4d,hall_size,1)
      fake_b_raw = fake_b_raw.detach()
      #fake_b_raw = fake_b_raw
      fake_b_image = input.clone()#元イメージを置き換える
      fake_b_image[:,:,center-d:center+d,center-d:center+d] = fake_b_raw[:,:,center-d:center+d,center-d:center+d]
      #テストエラーを作成する

      fake_b_image_raw_4d = torch.cat((fake_b_raw,mask),1) #catはメインループ内で許可
      
      reconstruct_error = criterionMSE(torch.masked_select(fake_b_raw, mask_channel_3d_b),torch.masked_select(real_a_image_4d[:,0:3,:,:], mask_channel_3d_b))# 生成画像とオリジナルの差
      #fake_D_predを用いたエラー
      Mdpos_x = center
      Mdpos_y = center#12/31 テストのマスクは固定で中央にしてみる

      if flag_local:
        pred_realD_Local  =  netD_Local.forwardWithTrim(real_a_image_4d.detach(),_xpos = Mdpos_x,_ypos = Mdpos_y,trim_size = Local_Window,batchSize = opt.batchSize)
        pred_fakeD_Local = netD_Local.forwardWithTrimCover(fake_b_image_raw_4d.detach(),_xpos = Mdpos_x,_ypos = Mdpos_y,trim_size = Local_Window,_input_real = real_a_image_4d,hole_size = hall_size,batchSize = opt.batchSize) #pred_falke=D(C(x,Mc),Mc)
      if flag_global:
        pred_fakeD_Global = netD_Global.forwardWithCover(fake_b_image_raw_4d,_input_real = real_a_image_4d,hole_size = hall_size) #pred_falke=D(C(x,Mc),Mc)
        pred_realD_Global =  netD_Global.forward(real_a_image_4d.detach())
      if flag_edge:
        pred_realD_Edge  = netD_Edge.forwardWithTrim(real_a_image_4d.detach(),_xpos = Mdpos_x,_ypos = Mdpos_y,trim_size = Local_Window,batchSize = opt.batchSize)
        pred_fakeD_Edge  = netD_Edge.forwardWithTrimCover(fake_b_image_raw_4d.detach(),_xpos = Mdpos_x,_ypos = Mdpos_y,trim_size = Local_Window,_input_real = real_a_image_4d,hole_size = hall_size,batchSize = opt.batchSize) #pred_falke=D(C(x,Mc),Mc)

      if (flag_global == True) and (flag_local == True) and (flag_edge == True):
        #Concatを使って繋げる
        pred_realD = net_Concat(pred_realD_Global,pred_realD_Local)
        pred_fakeD = net_Concat(pred_fakeD_Global,pred_fakeD_Local)
      #〇〇
      if (flag_global == True) and (flag_local == True) and (flag_edge == False):
        #Concatを使って繋げる
        pred_realD = net_Concat(pred_realD_Global,pred_realD_Local)
        pred_fakeD = net_Concat(pred_fakeD_Global,pred_fakeD_Local)

      #〇×
      if (flag_global == True) and (flag_local == False):
        pred_realD = pred_realD_Global
        pred_fakeD = pred_fakeD_Global
      #×〇
      if (flag_global == False) and (flag_local == True):
        pred_realD = pred_realD_Global
        pred_fakeD = pred_fakeD_Global

      false_label_tensor = Variable(torch.LongTensor())
      false_label_tensor  = torch.zeros(1,1)
      true_label_tensor = Variable(torch.LongTensor())
      true_label_tensor  = torch.ones(1,1)


      loss_d_realD = criterionBCE(pred_realD, true_label_tensor)
      loss_d_fakeD = criterionBCE(pred_fakeD, false_label_tensor) #ニセモノ-ホンモノをニセモノと判断させたいのでfalse

      #プロット用にGlobalとLocalのLossを上とは別に個別に導出する
      if (flag_global == True):
        pred_realD_Global = net_Concat1.forward1(pred_realD_Global)
        pred_fakeD_Global = net_Concat1.forward1(pred_fakeD_Global)
        pred_realD_Local = net_Concat1.forward1(pred_realD_Local)
        pred_fakeD_Local = net_Concat1.forward1(pred_fakeD_Local)

        loss_d_realD_Global = criterionBCE(pred_realD_Global, true_label_tensor)
        loss_d_fakeD_Global = criterionBCE(pred_fakeD_Global, false_label_tensor)
      if (flag_local == True):
        loss_d_realD_Local  = criterionBCE(pred_realD_Local, true_label_tensor)
        loss_d_fakeD_Local  = criterionBCE(pred_fakeD_Local, false_label_tensor)

      if (flag_edge == True):
        pred_fakeD_Edge = net_Concat1.forward1(pred_fakeD_Edge)
        pred_realD_Edge = net_Concat1.forward1(pred_realD_Edge)
        loss_d_realD_Edge = criterionBCE(pred_realD_Edge, true_label_tensor)
        loss_d_fakeD_Edge = criterionBCE(pred_fakeD_Edge, true_label_tensor)

    #2つのロスの足し合わせ
      loss_d = 0
      if mode == 1:
        loss_d = loss_d_realD + loss_d_fakeD 
      if mode == 2:
        if (flag_edge == True):
          loss_d = (loss_d_fakeD + loss_d_realD + loss_d_fakeD_Edge + loss_d_realD_Edge) * disc_weight
        else:
          loss_d = (loss_d_fakeD + loss_d_realD) * disc_weight
    #backward

      #最終的なロスの導出
      test_loss_g = reconstruct_error
      if (mode == 2):

        test_loss_g += loss_d_fakeD
        if (flag_edge == True):
          test_loss_g += loss_d_fakeD_Edge 
      test_loss_d = loss_d

      fake_b_image = fake_b_image.cpu()
      out_img = fake_b_image.data[0]


      #生成の結果のプロット       
      Plot2Image(fake_b_image,TestFakeB_dir_,'/'+ str(epoch)+'_'+image_name)        
      Plot2Image(edge_detection( fake_b_image,False),TestFakeB_Edge_dir_,'/'+ str(epoch) +'_'+image_name)        




      loss_g_avg += test_loss_g
      if(mode == 1 or mode == 2):
        loss_d_avg += test_loss_d
        loss_dg_r_avg += loss_d_realD_Global
        loss_dg_f_avg += loss_d_fakeD_Global
        loss_dl_r_avg += loss_d_realD_Local
        loss_dl_f_avg += loss_d_fakeD_Local
        loss_de_r_avg += loss_d_realD_Edge
        loss_de_f_avg += loss_d_fakeD_Edge
      #ロスの平均の導出
      if(iteration ==  (max_test_dataset_num)):
        loss_g_avg = loss_g_avg / iteration
        if(mode == 1 or mode == 2):
          loss_d_avg = loss_d_avg / iteration
        result_list.append('{:.4g}'.format(loss_g_avg))
        if(mode == 1 or mode == 2):
          result_list.append('{:.4g}'.format(loss_d_avg))
        if(mode == 1 or mode == 2):
          loss_d_avg = 0
        loss_g_avg = 0
        #フラグが有効なら個別のロスを記録する
        if(each_loss_plot_flag and mode != 0):
          loss_dg_r_avg = loss_dg_r_avg / iteration 
          loss_dg_f_avg = loss_dg_f_avg / iteration 
          loss_dl_r_avg = loss_dl_r_avg / iteration
          loss_dl_f_avg = loss_dl_f_avg / iteration
          loss_de_r_avg = loss_de_r_avg / iteration 
          loss_de_f_avg = loss_de_f_avg / iteration 

          result_list.append(('{:.4g}'.format(loss_d_realD_Global)))
          result_list.append(('{:.4g}'.format(loss_d_fakeD_Global)))
          result_list.append(('{:.4g}'.format(loss_d_realD_Local)))
          result_list.append(('{:.4g}'.format(loss_d_fakeD_Local)))
          result_list.append(('{:.4g}'.format(loss_d_realD_Edge)))
          result_list.append(('{:.4g}'.format(loss_d_fakeD_Edge)))
        else:
          for i in range(6):
            result_list.append(0)
            
        if mode == 0:
          result_list.append(0)

      loss_dg_r_avg = 0
      loss_dg_f_avg = 0
      loss_dl_r_avg = 0
      loss_dl_f_avg = 0
      loss_de_r_avg = 0
      loss_de_f_avg = 0

      iteration = iteration + 1

#リザルト関連の処理
result_dir = 'Results' 
Time_dir =  '/'+ str(start_date) + '-' + str(start_time.hour) +'-' + str(start_time.minute)  #時刻
Image_Train_dir = '/Image_Train'
Model_dir = '/Models'
Loss_dir = '/Losses'
Image_Test_dir = '/Image_Test'

#Image_Trainディレクトリ
RealA_dir = '/RealA'
FakeB_dir = '/FakeB'
FakeB_Raw_dir = '/FaleB_Raw'
FakeB_Edge_dir = '/FakeB_Edge'

#Image_Testディレクトリ(↑と同一)
RealA_dir = '/RealA'
FakeB_dir = '/FakeB'
FakeB_Raw_dir = '/FaleB_Raw'
FakeB_Edge_dir = '/FakeB_Edge'

#Modelディレクトリ
netG_dir = '/netG'
netDg_dir = '/netDg'
netDl_dir = '/netDl'
netDe_dir = '/netDe'

#末尾に_付きがトータルのディレクトリ
TrainRealA_dir_ = result_dir+ Time_dir + Image_Train_dir + RealA_dir
TrainFakeB_dir_ = result_dir+ Time_dir + Image_Train_dir + FakeB_dir
TrainFakeB_Raw_dir_ = result_dir+ Time_dir + Image_Train_dir + FakeB_Raw_dir
TrainFakeB_Edge_dir_ = result_dir+ Time_dir + Image_Train_dir + FakeB_Edge_dir

TestRealA_dir_ = result_dir+ Time_dir + Image_Test_dir + RealA_dir
TestFakeB_dir_ = result_dir+ Time_dir + Image_Test_dir + FakeB_dir
TestFakeB_Raw_dir_ = result_dir+ Time_dir + Image_Test_dir + FakeB_Raw_dir
TestFakeB_Edge_dir_ = result_dir+ Time_dir + Image_Test_dir + FakeB_Edge_dir

Model_netG_dir_  = result_dir+ Time_dir + Model_dir + netG_dir
Model_netDg_dir_ = result_dir+ Time_dir + Model_dir + netDg_dir
Model_netDl_dir_ = result_dir+ Time_dir + Model_dir + netDl_dir
Model_netDe_dir_ = result_dir+ Time_dir + Model_dir + netDe_dir

Loss_dir_ = result_dir+ Time_dir + Loss_dir

os.makedirs(TrainRealA_dir_  ,exist_ok=True)
os.makedirs(TrainFakeB_dir_   ,exist_ok=True)
os.makedirs(TrainFakeB_Raw_dir_  ,exist_ok=True)
os.makedirs(TrainFakeB_Edge_dir_  ,exist_ok=True)

os.makedirs(TestRealA_dir_  ,exist_ok=True)
os.makedirs(TestFakeB_dir_  ,exist_ok=True)
os.makedirs(TestFakeB_Raw_dir_  ,exist_ok=True)
os.makedirs(TestFakeB_Edge_dir_  ,exist_ok=True)

os.makedirs(Model_netG_dir_  ,exist_ok=True)
os.makedirs(Model_netDg_dir_ ,exist_ok=True)
os.makedirs(Model_netDl_dir_  ,exist_ok=True)
os.makedirs(Model_netDe_dir_  ,exist_ok=True)

os.makedirs(Loss_dir_  ,exist_ok=True)

#↑の通りにディレクトリを作成する

#プロパティのログを出力する(モード,エポック数など)



#確認のための画像出力メソッド
def tensor_plot2image(__input,name,iteration=1,mode=0):
  if mode == 0:
    mode_dir = 'testing_output\\'
  elif mode == 1:
    mode_dir = 'testing_output_disc\\'
  else :
    mode_dir = 'testing_output_total\\'


  dirname = mode_dir + str(start_date) + '-' + str(start_time.hour) + '-' + str(start_time.minute) + '-' + str(start_time.second) 
  
  if not os.path.exists(dirname):
    os.mkdir(dirname)
  path = os.getcwd() + '\\' + dirname + '\\'
  vutils.save_image(__input.detach(), path + name + '.jpg')

  #print('saved testing image')

def Plot2Image(__input,__dir,__name):
  vutils.save_image(__input.detach(), __dir + '/' + __name + '.jpg')

def SaveModel(epoch,mode=0):
  if mode != 1:
    net_g_model_out_path =  Model_netG_dir_ + "/netG_{}.pth".format(epoch)
    torch.save(netG, net_g_model_out_path)
  if mode != 0:
    net_dg_model_out_path =  Model_netDg_dir_ + "/netDg_{}.pth".format(epoch)
    torch.save(netD_Global, net_dg_model_out_path)
    net_dl_model_out_path =  Model_netDl_dir_ + "/netDl_{}.pth".format(epoch)
    torch.save(netD_Local, net_dl_model_out_path)
    net_de_model_out_path =  Model_netDe_dir_ + "/netDe_{}.pth".format(epoch)
    torch.save(netD_Edge, net_de_model_out_path)


def checkpoint(epoch,mode=0):
  if mode == 0:
    mode_dir = 'checkpoint_gene'
  elif mode == 1:
    mode_dir = 'checkpoint_disc'
  else :
    mode_dir = 'checkpoint_total' 

  dirname = mode_dir + '\\' + str(start_date) + '-' + str(start_time.hour) + '-' + str(start_time.minute) + '-' + str(start_time.second) 
  if not os.path.exists(dirname):
    os.mkdir(dirname)
  path = os.getcwd() + '\\' + dirname
  if mode != 1:
    net_g_model_out_path = "{}/netG_model_epoch_{}_mode_{}.pth".format(path,epoch,mode)
    torch.save(netG, net_g_model_out_path)
  if mode != 0:
    net_dg_model_out_path = "{}/netDg_model_epoch_{}_mode_{}.pth".format(path,epoch,mode)
    torch.save(netD_Global, net_dg_model_out_path)
    net_dl_model_out_path = "{}/netDl_model_epoch_{}_mode_{}.pth".format(path,epoch,mode)
    torch.save(netD_Local, net_dl_model_out_path)
    net_de_model_out_path = "{}/netDe_model_epoch_{}_mode_{}.pth".format(path,epoch,mode)
    torch.save(netD_Edge, net_de_model_out_path)
  #print("Checkpoint saved to {}".format("checkpoint" + opt.dataset))

gene_only_epoch = 10
disc_only_epoch = 2
total_epoch = 500
#Test = False
#使用する既存のモデルがある場合はここでloadする



def PlotError():
  with open(Loss_dir_ + '/loss_log_result.csv', 'a') as f:
    #writer = csv.writer(f)
    #string = ''
    #f.write('')
    result_str = ",".join(map(str,result_list)) #カンマ区切りに直す
    f.write(result_str + '\n')
    #リストの中を取り出す
    #for s in result_list:
    #  string += str(s)
    #  string += ','
    #writer.writerows(string)
  #リストを空にする
  result_list.clear()

#最初のカラムを作成する
result_list.append("Epoch[/n]")
result_list.append("Train_Loss_G")
result_list.append("Train_Loss_D")
result_list.append("Train_Loss_Dg_R")
result_list.append("Train_Loss_Dg_F")
result_list.append("Train_Loss_Dl_R")
result_list.append("Train_Loss_Dl_F")
result_list.append("Train_Loss_De_R")
result_list.append("Train_Loss_De_F")
result_list.append("Test_Loss_G")
result_list.append("Test_Loss_D")
result_list.append("Test_Loss_Dg_R")
result_list.append("Test_Loss_Dg_F")
result_list.append("Test_Loss_Dl_R")
result_list.append("Test_Loss_Dl_F")
result_list.append("Test_Loss_De_R")
result_list.append("Test_Loss_De_F")
PlotError()



for epoch in range(total_epoch):
#discriminatorのtrain
  train(epoch,mode=2)#両方
  test(epoch,2)
  if(epoch % 5 == 0):
    SaveModel(epoch,2)

  PlotError()


for epoch in range(1, gene_only_epoch + 1):
#discriminatorのtrain

  train(epoch,mode=0)#Discriminatorのみ
  test(epoch,0)
  if(epoch % 5 == 0):
    SaveModel(epoch,0)

  PlotError()


for epoch in range(1, disc_only_epoch + 1):
#discriminatorのtrain
  #netG = torch.load("checkpoint/testing_modelG_15.pth")
  train(epoch,mode=1)#Discriminatorのみ
  test(epoch,1)
#  checkpoint(epoch,1)
  if(epoch % 5 == 0):
    SaveModel(epoch,1)
  PlotError()
#if Test==True:
  #test(1)













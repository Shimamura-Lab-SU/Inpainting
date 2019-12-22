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
from   networks               import define_G, GANLoss, print_network, define_D_Edge,define_D_Global,define_D_Local,define_Concat
from   data                   import get_training_set, get_test_set
import torch.backends.cudnn as cudnn
from   util                   import save_img
from   function import Set_Md, Set_Masks
from tqdm import tqdm


import random
import time
import datetime

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
parser.add_argument('--lr', type=float, default=0.0004, help='Learning Rate. Default=0.002')
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
if opt.cuda:
  torch.cuda.manual_seed(opt.seed)

print('===> Loading datasets')
root_path            = "dataset/"
train_set            = get_training_set(root_path + opt.dataset)
test_set             = get_test_set(root_path + opt.dataset)
training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=False)
testing_data_loader  = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=opt.testBatchSize, shuffle=False)

max_dataset_num = 1500#データセットの数

train_set.image_filenames = train_set.image_filenames[:max_dataset_num]
test_set.image_filenames = test_set.image_filenames[:max_dataset_num]

print('===> Building model')

#3つのタスクのそれぞれのエポック

#先ず学習済みのGeneratorを読み込んで入れる
#netG = torch.load(opt.G_model)
disc_input_nc = 4
disc_outpuc_nc = 1024
#netG = torch.load("checkpoint/testing_modelG_50.pth")
netG = define_G(4, 3, opt.ngf, 'batch', False, [0])
netD_Global = define_D_Global(disc_input_nc , disc_outpuc_nc, opt.ndf,  [0])
#netD_Global = torch.load("checkpoint/testing_modelDg_4.pth")  
#netD_Global = torch.load("checkpoint/testing_modelDg_10.pth")
netD_Local   = define_D_Local(disc_input_nc , disc_outpuc_nc, opt.ndf,  [0])
netD_Edge     = define_D_Edge(disc_input_nc , disc_outpuc_nc, opt.ndf,  [0])
net_Concat = define_Concat(2048,1,[0])

criterionGAN = GANLoss()
criterionL1 = nn.L1Loss()
criterionMSE = nn.MSELoss()
criterionBCE = nn.BCELoss()#Discriminatorに使うため新設
# setup optimizer
optimizerG = optim.Adadelta(netG.parameters(), lr=opt.lr)
optimizerD_Global = optim.Adadelta(netD_Global.parameters(), lr=opt.lr)
optimizerD_Local = optim.Adadelta(netD_Local.parameters(), lr=opt.lr)
optimizerD_Edge = optim.Adadelta(netD_Edge.parameters(), lr=opt.lr)

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
  netG = netG.cuda()
  net_Concat = net_Concat.cuda()
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
mask_channel_float = Set_Masks(image_size,center,center,hall_size) #古い宣言
mask_channel_boolen = Set_Masks(image_size,center,center,hall_size,bool)
##Md(RandomMask)の定義


#Mdの場所のためのシード固定
seed = random.seed(1297)

#マスク、ラベルテンソルのgpuセット
#if opt.cuda:
#  mask_channel_boolen = mask_channel_boolen.cuda()
#  mask_channel_float = mask_channel_float.cuda()
#  random_mask_boolen_64 = mask_channel_boolen.cuda() #代入間違ってるけど大丈夫か
#  random_mask_float_64 = mask_channel_float.cuda()
#  random_mask_boolen_128 = mask_channel_boolen.cuda()
#  random_mask_float_128 = mask_channel_float.cuda()
#  true_label_tensor = true_label_tensor.cuda()
#  false_label_tensor = false_label_tensor.cuda()
#  white_channel_float_128 = white_channel_float_128.cuda()
start_date = datetime.date.today()
start_time = datetime.datetime.now()



def train(epoch,mode=0):

  center = math.floor(image_size / 2)
  d = math.floor(Local_Window / 4) #trim(LocalDiscriminator用の窓)
  d2 = math.floor(Local_Window / 2) #L1Loss用の純粋な生成画像と同じサイズの窓用,所謂Mc

  mask_channel_boolen = Set_Masks(image_size,center,center,hall_size,bool)
  mask_channel_float = Set_Masks(image_size,center,center,hall_size)

  epoch_start_time = time.time()
  #loss_plot_array = np.zeros(int(opt.batchSize / max_dataset_num),4) # 1エポックごとにロスを書き込む
  loss_plot_array = np.empty((int(max_dataset_num / opt.batchSize),4))
  #0..OnlyGenerator
  #1..OnlyDiscriminator
  #2..Both 
  flag_global = True
  flag_local  = True
  flag_edge   = False

  mask_channel_3d_b = torch.cat((mask_channel_boolen,mask_channel_boolen,mask_channel_boolen),1)
  mask_channel_3d_b = mask_channel_3d_b.cuda()
  #Generatorの学習タスク
  for iteration, batch in tqdm(enumerate(training_data_loader, 1)):


    #####################################################################
    #ランダムマスクの作成
    #####################################################################

    #Mdの位置
    Mdpos_x,Mdpos_y = Set_Md(seed)
    #Mdを↑の位置に当てはめる
    
    random_mask_float_64 = Set_Masks(image_size,Mdpos_x,Mdpos_y,hall_size,torch.float32)
    random_mask_boolen_64 = Set_Masks(image_size,Mdpos_x,Mdpos_y,hall_size,bool)
    random_mask_float_128 = Set_Masks(image_size,Mdpos_x,Mdpos_y,Local_Window,torch.float32)
    random_mask_boolen_128 = Set_Masks(image_size,Mdpos_x,Mdpos_y,Local_Window,bool)

    #if opt.cuda:
    #  random_mask_float_64 =  random_mask_float_64.cuda()
    #  random_mask_boolen_64 = random_mask_boolen_64.cuda()
    #  random_mask_float_128 = random_mask_float_128.cuda()
    #  random_mask_boolen_128 = random_mask_boolen_128.cuda()  

    #####################################################################
    #イメージテンソルの定義
    #####################################################################
    real_a_image_cpu = batch[1]#batch[1]が原画像そのまんま
    real_a_image.data.resize_(real_a_image_cpu.size()).copy_(real_a_image_cpu)#realAへのデータ流し込み
    #real_a_image = batch[1]
    #####################################################################
    #先ずGeneratorを起動して補完モデルを行う
    #####################################################################
    #fake_start_imageは単一画像中の平均画素で埋められている


    #fake_start_image = torch.clone(real_a_image) #cp ←cpu

    #for i in range(0, opt.batchSize):#中心中心
    #  fake_start_image[i][0] = torch.mean(real_a_image[i][0])
    #  fake_start_image[i][1] = torch.mean(real_a_image[i][1])
    #  fake_start_image[i][2] = torch.mean(real_a_image[i][2])

    #fake_start_image2を穴のサイズにトリムしたもの
    #fake_start_image2 = fake_start_image[:][:][0:hall_size][0:hall_size]
    #fake_start_image2.resize_(opt.batchSize,opt.input_nc,hall_size,hall_size)

    #12/10real_b_imageはreal_aの穴に平均画素を詰めたもの
    #center = math.floor(image_size / 2)
    #d = math.floor(Local_Window / 4) #4(窓サイズの1/4)
    #real_b_image = real_a_image.clone()    
    #real_b_image[:,:,center - d:center+d,center - d:center+d] = fake_start_image[:,:,center - d:center+d,center - d:center+d]

    #####################################################################
    #Global,LocalDiscriminatorを走らせる
    #####################################################################
    real_a_image_4d = torch.cat((real_a_image,random_mask_float_64),1)
    real_a_image_4d = real_a_image_4d.cuda() 
    mask_channel_float = mask_channel_float.cuda()#どうしても必要なためcudaに入れる


    #real_b_image_4d = torch.cat((real_b_image,mask_channel_float),1)#これはreal_aから作れないか？
    #real_b_image_4d = real_b_image_4d.cuda()

    #1stDiscriminator
    if mode==1 or mode==2:
      #マスクとrealbの結合

      #12/17optimizerをzero_gradする
      #128*128の窓を作成 (あとで裏に回れるようにする)
      #真画像とマスクの結合..4次元に
      fake_b_image_raw = netG.forwardWithMasking(real_a_image_4d,hall_size) # C(x,Mc) #ここをnetGの機能にする
      
      fake_b_image_raw_4d = torch.cat((fake_b_image_raw,mask_channel_float),1) #catはメインループ内で許可
      fake_b_image_raw_4d = fake_b_image_raw_4d.cuda()


      if flag_global:
        pred_fakeD_Global = netD_Global.forwardWithCover(fake_b_image_raw_4d.detach(),_input_real = real_a_image_4d,hole_size = hall_size) #pred_falke=D(C(x,Mc),Mc)
        pred_realD_Global =  netD_Global.forward(real_a_image_4d.detach())
      if flag_local:
        pred_realD_Local  =  netD_Local.forwardWithTrim(real_a_image_4d.detach(),_xpos = Mdpos_x,_ypos = Mdpos_y,trim_size = Local_Window)
        pred_fakeD_Local = netD_Local.forwardWithTrimCover(fake_b_image_raw_4d.detach(),_xpos = Mdpos_x,_ypos = Mdpos_y,trim_size = Local_Window,_input_real = real_a_image_4d,hole_size = hall_size) #pred_falke=D(C(x,Mc),Mc)
      if flag_edge:
        #real_a_image_edge.datach
        pred_realD_Edge  = netD_Edge.forwardWithTrim(real_a_image_4d.detach(),_xpos = Mdpos_x,_ypos = Mdpos_y,trim_size = Local_Window)
        #real_a_image_edge.datach #エッジ処理はネットワーク内でさせる
        pred_fakeD_Edge  = netD_Edge.forwardWithTrimCover(fake_b_image_raw_4d.detach(),_xpos = Mdpos_x,_ypos = Mdpos_y,trim_size = Local_Window,_input_real = real_a_image_4d,hole_size = hall_size) #pred_falke=D(C(x,Mc),Mc)

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

      true_label_tensor = true_label_tensor.cuda()
      false_label_tensor = false_label_tensor.cuda()
      #loss_d = loss_d_realG_Global + loss_d_fakeG_Local
      loss_d_realD = criterionBCE(pred_realD, true_label_tensor)
      loss_d_fakeD = criterionBCE(pred_fakeD, false_label_tensor) #ニセモノ-ホンモノをニセモノと判断させたいのでfalse
      
      if (flag_edge == True):
        loss_d_realD_Edge = criterionBCE(pred_realD_Edge, true_label_tensor)
        loss_d_fakeD_Edge = criterionBCE(pred_fakeD_Edge, true_label_tensor)

      #loss_d_realG_Local = criterionBCE(pred_realD_Local, true_label_tensor)
      #loss_d_fakeG_Local = criterionBCE(pred_fakeD_Local, false_label_tensor) #ニセモノ-ホンモノをニセモノと判断させたいのでfalse

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
      fake_b_image_raw = netG.forwardWithMasking(real_a_image_4d,hall_size) # C(x,Mc)
      #fake_b_image = real_a_image.clone()
      #fake_b_image[:,:,center-d:center+d,center-d:center+d] = fake_b_image_raw[:,:,center-d:center+d,center-d:center+d] 
      fake_b_image_raw_4d = torch.cat((fake_b_image_raw,mask_channel_float),1) #catはメインループ内で許可
      fake_b_image_raw_4d = fake_b_image_raw_4d.cuda()

      #2ndDiscriminator
      if mode == 2:
        if flag_global: 
          pred_fakeD_Global = netD_Global.forwardWithCover(fake_b_image_raw_4d.detach(),_input_real = real_a_image_4d,hole_size = hall_size) #pred_falke=D(C(x,Mc),Mc)
        if flag_local:
          pred_fakeD_Local = netD_Local.forwardWithTrimCover(fake_b_image_raw_4d.detach(),_xpos = Mdpos_x,_ypos = Mdpos_y,trim_size = Local_Window,_input_real = real_a_image_4d,hole_size = hall_size) #pred_falke=D(C(x,Mc),Mc)
        if flag_edge:
          pred_fakeD_Edge = netD_Edge.forwardWithTrimCover(fake_b_image_raw_4d.detach(),_xpos = Mdpos_x,_ypos = Mdpos_y,trim_size = Local_Window,_input_real = real_a_image_4d,hole_size = hall_size) #pred_falke=D(C(x,Mc),Mc)
          
        #pred_fakeは偽生成画像を入力としたときの尤度テンソル
        #〇〇〇
        if (flag_global == True) and (flag_local == True) and (flag_edge == True):
          pred_realD = net_Concat(pred_realD_Global,pred_realD_Local)#エッジはConcatしない
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



		  #reconstructError
      #fake_b_image_masked =  #GPU回避不可能？
      #real_a_image_masked =  #1次元で(61440)出てくるので..
      reconstruct_error = criterionMSE(torch.masked_select(fake_b_image_raw, mask_channel_3d_b),torch.masked_select(real_a_image_4d[:,0:3,:,:], mask_channel_3d_b))# 生成画像とオリジナルの差
      #fake_D_predを用いたエラー
      loss_g = reconstruct_error
      if mode == 2:
        loss_d_fakeD = disc_weight * criterionBCE(pred_fakeD, true_label_tensor) #ニセモノ-ホンモノをニセモノと判断させたいのでfalse
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



    #####################################################################
    #ログの作成、画像の出力
    #####################################################################
    if mode == 0 or mode == 2:
      #print("===> Epoch[{}]({}/{}):		Loss_G: {:.4f}".format(epoch, iteration, len(training_data_loader), loss_g.item()  ))
      if(epoch % 10 == 9):
        if(iteration <= 10):#10っ回に1回は10倍サンプルを吐く
          tensor_plot2image(fake_b_image_raw,'fakeC_Raw_Last_Epoch{}X_{}'.format(iteration,epoch),iteration,mode)
          tensor_plot2image(fake_b_image,'fakeC_Last_Epoch{}X_{}'.format(iteration,epoch),iteration,mode)
      else:
        if(iteration == 1):
          tensor_plot2image(fake_b_image_raw,'fakeC_Raw_Last_Epoch_{}'.format(epoch),iteration,mode)
          tensor_plot2image(fake_b_image,'fakeC_Last_Epoch_{}'.format(epoch),iteration,mode)

    #if mode == 1 or mode == 2:
      #後でGlobalとLocalで同時に出すことも検討
      #print("===> Epoch[{}]({}/{}): loss_d: {:.4f}".format(epoch, iteration, len(training_data_loader),  loss_d.item()))
    if iteration == len(training_data_loader):
      if mode == 0:
        mode_dir = 'loss_output_gene\\'
      elif mode == 1:
        mode_dir = 'loss_output_disc\\'
      else :
        mode_dir = 'loss_output_total\\'
      dirname = mode_dir + str(start_date) + '-' + str(start_time.hour) + '-' + str(start_time.minute) + '-' + str(start_time.second) 

      if not os.path.exists(dirname):
        os.mkdir(dirname)
      path = os.getcwd() + '\\' + dirname + '\\'
      with open(path + 'loss_log.csv', 'a') as f:
        writer = csv.writer(f)
        writer.writerows(loss_plot_array)





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
    net_g_model_out_path = "{}/netG_model_epoch_{}.pth".format(path,epoch)
    torch.save(netG, net_g_model_out_path)
  if mode != 0:
    net_dg_model_out_path = "{}/netDg_model_epoch_{}.pth".format(path,epoch)
    torch.save(netD_Global, net_dg_model_out_path)
    net_dl_model_out_path = "{}/netDl_model_epoch_{}.pth".format(path,epoch)
    torch.save(netD_Local, net_dl_model_out_path)
  print("Checkpoint saved to {}".format("checkpoint" + opt.dataset))


def checkpoint_total(epoch):
  if not os.path.exists("checkpoint"):
    os.mkdir("checkpoint")
  if not os.path.exists(os.path.join("checkpoint", opt.dataset)):
    os.mkdir(os.path.join("checkpoint", opt.dataset))
  net_g_model_out_path = "checkpoint/{}/Total_netG_model_epoch_{}.pth".format(opt.dataset, epoch)
  net_dg_model_out_path = "checkpoint/{}/Total_netDg_model_epoch_{}.pth".format(opt.dataset, epoch)
  net_dl_model_out_path = "checkpoint/{}/Total_netDl_model_epoch_{}.pth".format(opt.dataset, epoch)
  torch.save(netG, net_g_model_out_path)
  torch.save(netD_Global, net_dg_model_out_path)
  torch.save(netD_Local, net_dg_model_out_path)
  print("Checkpoint saved to {}".format("checkpoint" + opt.dataset))




gene_only_epoch = 50
disc_only_epoch = 0
total_epoch = 0

#使用する既存のモデルがある場合はここでloadする

for epoch in range(total_epoch):
#discriminatorのtrain
  train(epoch+1,mode=2)#両方
  checkpoint(epoch+1,2)




for epoch in range(1, gene_only_epoch + 1):
#discriminatorのtrain

  train(epoch,mode=0)#Discriminatorのみ
  checkpoint(epoch)



for epoch in range(1, disc_only_epoch + 1):
#discriminatorのtrain
  #netG = torch.load("checkpoint/testing_modelG_15.pth")
  train(epoch,mode=1)#Discriminatorのみ
  checkpoint(epoch)














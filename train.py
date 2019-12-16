from __future__ import print_function
import argparse
import os
from math import log10
import math
import cv2
from copy import copy

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
from   torch.autograd         import Variable
from   torch.utils.data       import DataLoader
from   networks               import define_G, GANLoss, print_network, define_D_Edge,define_D_Global,define_D_Local
from   data                   import get_training_set, get_test_set
import torch.backends.cudnn as cudnn
from   util                   import save_img

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
opt = parser.parse_args()
#自家製のハイパーパラメータ
Output_Each_Epoch = True #エポックの終了時に訓練結果を画像として出力するか否か
hall_size = 64 # 穴の大きさ(pic)
Local_Window = 128 #ローカルDiscが参照する範囲の大きさ(pic)
image_size = 256 #入力画像全体のサイズ

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

max_dataset_num = 1500
#trainsetの要素削減
train_set.image_filenames = train_set.image_filenames[:max_dataset_num]
test_set.image_filenames = test_set.image_filenames[:max_dataset_num]

print('===> Building model')
netG = define_G(4, 3, opt.ngf, 'batch', False, [0])
#NetDを3つ構築するのがよい
#netD = define_D(opt.input_nc + opt.output_nc, opt.ndf, 'batch', False, [0])
#そもそもいくつが入力なのか
disc_input_nc = 6
disc_outpuc_nc = 1024
netD_Global = define_D_Global(disc_input_nc , disc_outpuc_nc, opt.ndf,  [0])
netD_Local   = define_D_Local(disc_input_nc , disc_outpuc_nc, opt.ndf,  [0])
netD_Edge     = define_D_Edge(disc_input_nc , disc_outpuc_nc, opt.ndf,  [0])
#netD_Global = define_D_Global(hogehoge)
#netD_Local  = define_D_Local(hogehoge)
#netD_Edge   = define_D_Edge(hogehoge)


criterionGAN = GANLoss()
criterionL1 = nn.L1Loss()
criterionMSE = nn.MSELoss()


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

real_a0_image = torch.FloatTensor(opt.batchSize, opt.output_nc, 256, 256)
real_a_image = torch.FloatTensor(opt.batchSize, opt.output_nc, 256, 256)

#基本的なテンソルに使うShape
#tensor_shape = torch.tensor(opt.batchSize,3,image_size,image_size)
tensor_shape = torch.empty([opt.batchSize,3,image_size,image_size])
if opt.cuda:
  #netD = netD.cuda()
  netD_Global = netD_Global.cuda()
  netD_Local  = netD_Local.cuda()
  netD_Edge   = netD_Edge.cuda()

  netG = netG.cuda()
  criterionGAN = criterionGAN.cuda()
  criterionL1 = criterionL1.cuda()
  criterionMSE = criterionMSE.cuda()
  real_a_image = real_a_image.cuda()

real_a0_image = Variable(real_a0_image)
real_a_image = Variable(real_a_image)

#mask_channelは1*256*256の白黒マスク
center = math.floor(image_size / 2)
d = math.floor(Local_Window / 4) #4(窓サイズの1/4)
d2 = math.floor(Local_Window / 2) 

#black_channel = torch.tensor(dtype=bool)

black_channel_boolen = torch.full((opt.batchSize,1,image_size,image_size),False,dtype=bool)
white_channel_boolen = torch.full((opt.batchSize,1,hall_size,hall_size), True,dtype=bool)

black_channel_float = torch.full((opt.batchSize,1,image_size,image_size),False)
white_channel_float = torch.full((opt.batchSize,1,hall_size,hall_size), True)


mask_channel_float = black_channel_float.clone()
mask_channel_float[:,:,center - d:center+d,center - d:center+d] = white_channel_float


mask_channel_boolen = black_channel_boolen.clone()
mask_channel_boolen[:,:,center - d:center+d,center - d:center+d] = white_channel_boolen
if opt.cuda:
  mask_channel_boolen = mask_channel_boolen.cuda()
  mask_channel_float = mask_channel_float.cuda()

start_date = datetime.date.today()
start_time = datetime.datetime.now()
dirname = 'testing_output\\' + str(start_date) + '-' + str(start_time.hour) + '-' + str(start_time.minute) + '-' + str(start_time.second) 
os.mkdir(dirname)

#確認のための画像出力メソッド
def tensor_plot2image(__input,name,iteration=1):
  if(iteration == 1):
    path = os.getcwd() + '\\' + dirname + '\\'
    vutils.save_image(__input.detach(), path + name + '.jpg')
    print('saved testing image')



def train(epoch):
  #Generatorの学習タスク
  for iteration, batch in enumerate(training_data_loader, 1):
    real_a0_image_cpu, real_a_image_cpu = batch[0], batch[1]#batchは元画像？
    
    real_a0_image.data.resize_(real_a0_image_cpu.size()).copy_(real_a0_image_cpu)
    real_a_image.data.resize_(real_a_image_cpu.size()).copy_(real_a_image_cpu)

    #fake_start_imageは単一画像中の平均画素でfillされている
    fake_start_image = torch.clone(real_a_image)
    for i in range(0, opt.batchSize):
      fake_start_image[i][0] = torch.mean(real_a0_image[i][0])
      fake_start_image[i][1] = torch.mean(real_a0_image[i][1])
      fake_start_image[i][2] = torch.mean(real_a0_image[i][2])

    #fake_start_image2を穴のサイズにトリムしたもの
    fake_start_image2 = fake_start_image[:][:][0:hall_size][0:hall_size]
    fake_start_image2.resize_(opt.batchSize,opt.input_nc,hall_size,hall_size)

    #12/10real_cは真値の中央に平均画素を詰めたもの
    center = math.floor(image_size / 2)
    d = math.floor(Local_Window / 4) #4(窓サイズの1/4)
    real_b_image = real_a_image.clone() #参照渡しになっていたものを値渡しに変更    
    real_b_image[:,:,center - d:center+d,center - d:center+d] = fake_start_image[:,:,center - d:center+d,center - d:center+d]

  
    real_c_4d = torch.cat((real_b_image,mask_channel_float),1)
    #real_c_4d_b = torch.cat((real_b_image,mask_channel_boolen),1) #floatとboolはconcatできない


    #2回目のジェネレータ起動(forwardをするため)
    #fake_b_image_raw = torch.zeros(tensor_shape.size(),requires_grad=True) 
    fake_b_image_raw = netG.forward(real_c_4d)#穴画像
    #copyがデータを壊している
    fake_b_image_masked = copy(fake_b_image_raw[:,:,center-d:center+d,center-d:center+d]) 
    real_a_image_masked = copy(real_a_image[:,:,center-d:center+d,center-d:center+d]) 

    mask_channel_3d_b = torch.cat((mask_channel_boolen,mask_channel_boolen,mask_channel_boolen),1)

    fake_b_image_trim = torch.mul(fake_b_image_raw, mask_channel_3d_b) 
    real_a_image_trim = torch.mul(real_a_image, mask_channel_3d_b)

    #recept_tensor = torch.

    fake_b_image_masked = torch.masked_select(fake_b_image_raw, mask_channel_3d_b) 
    real_a_image_masked = torch.masked_select(real_a_image, mask_channel_3d_b) #1次元で(61440)出てくるので..

    reconstruct_error = criterionMSE(fake_b_image_masked,real_a_image_masked) # 生成画像とオリジナルの差
    #tensor_plot2image(fake_b_image_masked,'fake_b_image_masked',iteration)
    #tensor_plot2image(real_a_image_masked,'real_a_image_masked',iteration)



    #tensor_plot2image(fake_b_image,'fakeC_1',iteration)
    tensor_plot2image(fake_b_image_raw,'fakeC_Raw_1',iteration)
    tensor_plot2image(real_b_image,'realC_1',iteration)
    tensor_plot2image(real_a_image,'realb_1',iteration)

    #マスクをはりつけたもの
    fake_b_image = real_a_image.clone()
    fake_b_image[:,:,center-d:center+d,center-d:center+d] = fake_b_image_raw[:,:,center-d:center+d,center-d:center+d] 

    #loss_g = (loss_g1 + loss_g2) / 2 + loss_g_l2
    loss_g = reconstruct_error
    #loss_g.forward()

    loss_g.backward(retain_graph=True)
    #loss_g1.backward(retain_graph=True)
    #loss_g2.backward(retain_graph=True)
    #loss_g3.backward()
    optimizerG.step() # 動いてる
 #   print("===> Epoch[{}]({}/{}): Loss_D: {:.4f} Loss_G: {:.4f}".format(
#        epoch, iteration, len(training_data_loader), loss_d.item(), loss_g.item()))

    print("===> Epoch[{}]({}/{}):  Loss_G: {:.4f}".format(
       epoch, iteration, len(training_data_loader),  loss_g.item()))
    if(iteration == 1):
      tensor_plot2image(fake_b_image_raw,'fakeC_Raw_Last_Epoch_{}'.format(epoch),iteration)
      tensor_plot2image(fake_b_image,'fakeC_Last_Epoch_{}'.format(epoch),iteration)
      #vutils.save_image(fake_b_image_raw.detach(), '{}\\fake_b_image_Raw{:03d}.png'.format(os.getcwd() + '\\checkpoint_output', epoch,normalize=True, nrow=8))

  #Discriminatorの学習タスク
  

  #1epoch毎に出力してみる
  

#12/11テストタスクを全部↑に引っ越す
def test(epoch):
  avg_psnr = 0
  with torch.no_grad():
    for batch in testing_data_loader: 
      #input, target = Variable(batch[0]), Variable(batch[1])
      if opt.cuda:
        input = input.cuda()
        target = target.cuda()
#12/10　テストのためいったんなし
      #prediction = netG(input)
      #mse = criterionMSE(prediction, target)
      #psnr = 10 * log10(1 / mse.item())
      #avg_psnr += psnr
    #print("===> Avg. PSNR: {:.4f} dB".format(avg_psnr / len(testing_data_loader)))
     #チェックポイントの段階のモデルからアウトプットを作成する
  
    if opt.cuda:
      netG = netG.cuda()
      input = input.cuda()

      out = netG(input)
      out = out.cpu()
      out_img = out.data[0]  
      save_img(out_img, "checkpoint/{}/{}/{}".format(epoch,opt.dataset, image_name))


def checkpoint(epoch):
  if not os.path.exists("checkpoint"):
    os.mkdir("checkpoint")
  if not os.path.exists(os.path.join("checkpoint", opt.dataset)):
    os.mkdir(os.path.join("checkpoint", opt.dataset))
  net_g_model_out_path = "checkpoint/{}/netG_model_epoch_{}.pth".format(opt.dataset, epoch)
  net_dg_model_out_path = "checkpoint/{}/netDg_model_epoch_{}.pth".format(opt.dataset, epoch)
  net_dl_model_out_path = "checkpoint/{}/netDl_model_epoch_{}.pth".format(opt.dataset, epoch)
  net_de_model_out_path = "checkpoint/{}/netDe_model_epoch_{}.pth".format(opt.dataset, epoch)
  torch.save(netG, net_g_model_out_path)
  torch.save(netD_Global, net_dg_model_out_path)
  torch.save(netD_Local, net_dl_model_out_path)
#  torch.save(netD_Edge, net_d_model_out_path)
  print("Checkpoint saved to {}".format("checkpoint" + opt.dataset))



 


for epoch in range(1, opt.nEpochs + 1):
  train(epoch)
  #test(epoch)
  
  checkpoint(epoch)

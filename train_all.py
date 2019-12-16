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
from   function import Set_Md, Set_Masks

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

max_dataset_num = 1500 #データセットの数

train_set.image_filenames = train_set.image_filenames[:max_dataset_num]
test_set.image_filenames = test_set.image_filenames[:max_dataset_num]

print('===> Building model')

#3つのタスクのそれぞれのエポック



#先ず学習済みのGeneratorを読み込んで入れる
netG = torch.load(opt.G_model)

disc_input_nc = 4
disc_outpuc_nc = 1024
netD_Global = define_D_Global(disc_input_nc , disc_outpuc_nc, opt.ndf,  [0])
#netD_Global = torch.load("checkpoint/testing_modelDg_10.pth")
netD_Local   = define_D_Local(disc_input_nc , disc_outpuc_nc, opt.ndf,  [0])
netD_Edge     = define_D_Edge(disc_input_nc , disc_outpuc_nc, opt.ndf,  [0])


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

real_a_image = torch.FloatTensor(opt.batchSize, opt.output_nc, 256, 256)

if opt.cuda:
  #
  netD_Global = netD_Global.cuda()
  netD_Local  = netD_Local.cuda()
  netD_Edge   = netD_Edge.cuda()
  netG = netG.cuda()
  #
  criterionGAN = criterionGAN.cuda()
  criterionL1 = criterionL1.cuda()
  criterionMSE = criterionMSE.cuda()
  criterionBCE = criterionBCE.cuda()
  #
  real_a_image = real_a_image.cuda()

real_a_image = Variable(real_a_image)


##白、黒チャネルの定義
black_channel_boolen = torch.full((opt.batchSize,1,image_size,image_size),False,dtype=bool)
white_channel_boolen = torch.full((opt.batchSize,1,image_size,image_size), True,dtype=bool)
black_channel_float = torch.full((opt.batchSize,1,image_size,image_size),False)
white_channel_float = torch.full((opt.batchSize,1,image_size,image_size), True)
white_channel_float_128 = torch.full((opt.batchSize,1,128,128), True)


##Mc(inputMask)の定義
mask_channel_float = black_channel_float.clone() #mask_channel=Mcに相当,Gが穴を開けた位置(必ず中央)
mask_channel_float = Set_Masks(image_size,center,center,hall_size)
mask_channel_boolen = black_channel_boolen.clone()
mask_channel_boolen = Set_Masks(image_size,center,center,hall_size,bool)
##Md(RandomMask)の定義
random_mask_float_64 = black_channel_float.clone() #random_channel=Mdに相当,毎iterationランダムな位置に穴を空ける
random_mask_boolen_64 = black_channel_boolen.clone()
random_mask_float_128 = black_channel_float.clone() #random_channel=Mdに相当,毎iterationランダムな位置に穴を空ける
random_mask_boolen_128 = black_channel_boolen.clone()

false_label_tensor = Variable(torch.LongTensor())
false_label_tensor  = torch.zeros(opt.batchSize,2048,1,1)
true_label_tensor = Variable(torch.LongTensor())
true_label_tensor  = torch.ones(opt.batchSize,2048,1,1)

#Mdの場所のためのシード固定
seed = random.seed(1297)

#マスク、ラベルテンソルのgpuセット
if opt.cuda:
  mask_channel_boolen = mask_channel_boolen.cuda()
  mask_channel_float = mask_channel_float.cuda()
  random_mask_boolen_64 = mask_channel_boolen.cuda()
  random_mask_float_64 = mask_channel_float.cuda()
  random_mask_boolen_128 = mask_channel_boolen.cuda()
  random_mask_float_128 = mask_channel_float.cuda()
  true_label_tensor = true_label_tensor.cuda()
  false_label_tensor = false_label_tensor.cuda()
  white_channel_float_128 = white_channel_float_128.cuda()
start_date = datetime.date.today()
start_time = datetime.datetime.now()
dirname = 'testing_output_disc\\' + str(start_date) + '-' + str(start_time.hour) + '-' + str(start_time.minute) + '-' + str(start_time.second) 
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
 
    #Mdの位置
    Mdpos_x,Mdpos_y = Set_Md(seed)
    #Mdを↑の位置に当てはめる
    
    random_mask_float_64 = Set_Masks(image_size,Mdpos_x,Mdpos_y,hall_size,torch.float32)
    random_mask_boolen_64 = Set_Masks(image_size,Mdpos_x,Mdpos_y,hall_size,bool)
    random_mask_float_128 = Set_Masks(image_size,Mdpos_x,Mdpos_y,Local_Window,torch.float32)
    random_mask_boolen_128 = Set_Masks(image_size,Mdpos_x,Mdpos_y,Local_Window,bool)
    if opt.cuda:
      random_mask_float_64 =  random_mask_float_64.cuda()
      random_mask_boolen_64 = random_mask_boolen_64.cuda()
      random_mask_float_128 = random_mask_float_128.cuda()
      random_mask_boolen_128 = random_mask_boolen_128.cuda()  

    real_a_image_cpu = batch[1]#batch[1]が原画像そのまんま
    
    real_a_image.data.resize_(real_a_image_cpu.size()).copy_(real_a_image_cpu)#realAへのデータ流し込み
    #####################################################################
    #先ずGeneratorを起動して補完モデルを行う
    #####################################################################

    #fake_start_imageは単一画像中の平均画素で埋められている
    fake_start_image = torch.clone(real_a_image)
    for i in range(0, opt.batchSize):#中心中心
      fake_start_image[i][0] = torch.mean(real_a_image[i][0])
      fake_start_image[i][1] = torch.mean(real_a_image[i][1])
      fake_start_image[i][2] = torch.mean(real_a_image[i][2])

    #fake_start_image2を穴のサイズにトリムしたもの
    fake_start_image2 = fake_start_image[:][:][0:hall_size][0:hall_size]
    fake_start_image2.resize_(opt.batchSize,opt.input_nc,hall_size,hall_size)

    #12/10real_b_imageはreal_aの穴に平均画素を詰めたもの
    center = math.floor(image_size / 2)
    d = math.floor(Local_Window / 4) #4(窓サイズの1/4)
    real_b_image = real_a_image.clone()    
    real_b_image[:,:,center - d:center+d,center - d:center+d] = fake_start_image[:,:,center - d:center+d,center - d:center+d]

    #マスクとrealbの結合
    real_b_image_4d = torch.cat((real_b_image,mask_channel_float),1)
    #学習済みジェネレータで偽画像を作成
    fake_b_image_raw = netG(real_b_image_4d) # C(x,Mc)
   

    #####################################################################
    #Global,LocalDiscriminatorを走らせる
    #####################################################################

    #center..中央の定数
    center = math.floor(image_size / 2)
    d = math.floor(Local_Window / 4) #trim(LocalDiscriminator用の窓)
    d2 = math.floor(Local_Window / 2) #L1Loss用の純粋な生成画像と同じサイズの窓用,所謂Mc

    #fake_b_imageはfake_b_image_rawにreal_a_imageを埋めたもの
    fake_b_image = real_a_image.clone()
    fake_b_image[:,:,center-d:center+d,center-d:center+d] = fake_b_image_raw[:,:,center-d:center+d,center-d:center+d]

    fake_c_image = torch.Tensor(opt.batchSize,1,Local_Window,Local_Window)
    real_c_image = torch.Tensor(opt.batchSize,1,Local_Window,Local_Window)
    #128*128の窓を作成 (あとで裏に回れるようにする)
    fake_c_image = fake_b_image[:,:,Mdpos_x-d2:Mdpos_x+d2,Mdpos_y-d2:Mdpos_y+d2]#↑でランダムに決めた位置
    real_c_image = real_a_image[:,:,Mdpos_x-d2:Mdpos_x+d2,Mdpos_y-d2:Mdpos_y+d2]#↑でランダムに決めた位置

    #GlobalDiscriminatorの起動

    #真画像とマスクの結合..4次元に
    real_a_image_4d = torch.cat((real_a_image,random_mask_float_64),1)
    real_c_image_4d = torch.cat((real_c_image,white_channel_float_128),1) #LocalにもMdをかけるのか？←実質同じのため一旦All1のフィルタを入れる

    #pred_realは正しい画像を入力としたときの尤度テンソル(batchSize*1024*1*1)
    pred_realD_Global =  netD_Global.forward(real_a_image_4d)
    pred_realD_Local  =  netD_Local.forward(real_c_image_4d)

    #偽画像とマスクの結合..4次元に  
    fake_b_image_4d = torch.cat((fake_b_image,mask_channel_float),1)
    fake_c_image_4d = torch.cat((fake_c_image,white_channel_float_128),1)

    #pred_fakeは偽生成画像を入力としたときの尤度テンソル
    pred_fakeD_Global = netD_Global.forward(fake_b_image_4d) #pred_falke=D(C(x,Mc),Mc)
    pred_fakeD_Local = netD_Local.forward(fake_c_image_4d) #pred_falke=D(C(x,Mc),Mc)

    pred_realD = torch.cat((pred_realD_Global,pred_realD_Local),1)
    pred_fakeD = torch.cat((pred_fakeD_Global,pred_fakeD_Local),1)

    #loss_d = loss_d_realG_Global + loss_d_fakeG_Local
    loss_d_realD = criterionBCE(pred_realD, true_label_tensor)
    loss_d_fakeD = criterionBCE(pred_fakeD, false_label_tensor) #ニセモノ-ホンモノをニセモノと判断させたいのでfalse
    #loss_d_realG_Local = criterionBCE(pred_realD_Local, true_label_tensor)
    #loss_d_fakeG_Local = criterionBCE(pred_fakeD_Local, false_label_tensor) #ニセモノ-ホンモノをニセモノと判断させたいのでfalse

    #2つのロスの足し合わせ
    loss_d =  loss_d_realD + loss_d_fakeD 
    #backward
    loss_d.backward()
    

    #optimizerの更新
    optimizerD_Global.step()


    print("===> Epoch[{}]({}/{}):  Loss_D_Global: {:.4f}".format(
      epoch, iteration, len(training_data_loader),  loss_d.item()))



def total_train(epoch):
  #Generatorの学習タスク
  for iteration, batch in enumerate(training_data_loader, 1):
    #Mdの位置
    Mdpos_x,Mdpos_y = Set_Md(seed)
    #Mdを↑の位置に当てはめる
    
    random_mask_float_64 = Set_Masks(image_size,Mdpos_x,Mdpos_y,hall_size,torch.float32)
    random_mask_boolen_64 = Set_Masks(image_size,Mdpos_x,Mdpos_y,hall_size,bool)
    random_mask_float_128 = Set_Masks(image_size,Mdpos_x,Mdpos_y,Local_Window,torch.float32)
    random_mask_boolen_128 = Set_Masks(image_size,Mdpos_x,Mdpos_y,Local_Window,bool)
    if opt.cuda:
      random_mask_float_64 =  random_mask_float_64.cuda()
      random_mask_boolen_64 = random_mask_boolen_64.cuda()
      random_mask_float_128 = random_mask_float_128.cuda()
      random_mask_boolen_128 = random_mask_boolen_128.cuda()  

    real_a_image_cpu = batch[1]#batch[1]が原画像そのまんま
    
    real_a_image.data.resize_(real_a_image_cpu.size()).copy_(real_a_image_cpu)#realAへのデータ流し込み
    #####################################################################
    #先ずGeneratorを起動して補完モデルを行う
    #####################################################################

    #fake_start_imageは単一画像中の平均画素で埋められている
    fake_start_image = torch.clone(real_a_image)
    for i in range(0, opt.batchSize):#中心中心
      fake_start_image[i][0] = torch.mean(real_a_image[i][0])
      fake_start_image[i][1] = torch.mean(real_a_image[i][1])
      fake_start_image[i][2] = torch.mean(real_a_image[i][2])

    #fake_start_image2を穴のサイズにトリムしたもの
    fake_start_image2 = fake_start_image[:][:][0:hall_size][0:hall_size]
    fake_start_image2.resize_(opt.batchSize,opt.input_nc,hall_size,hall_size)

    #12/10real_b_imageはreal_aの穴に平均画素を詰めたもの
    center = math.floor(image_size / 2)
    d = math.floor(Local_Window / 4) #4(窓サイズの1/4)
    real_b_image = real_a_image.clone()    
    real_b_image[:,:,center - d:center+d,center - d:center+d] = fake_start_image[:,:,center - d:center+d,center - d:center+d]

    #マスクとrealbの結合
    real_b_image_4d = torch.cat((real_b_image,mask_channel_float),1)
    #学習済みジェネレータで偽画像を作成
    fake_b_image_raw = netG(real_b_image_4d) # C(x,Mc)
		#reconstructError
    mask_channel_3d_b = torch.cat((mask_channel_boolen,mask_channel_boolen,mask_channel_boolen),1)

    fake_b_image_masked = torch.masked_select(fake_b_image_raw, mask_channel_3d_b) 
    real_a_image_masked = torch.masked_select(real_a_image, mask_channel_3d_b) #1次元で(61440)出てくるので..


    reconstruct_error = criterionMSE(fake_b_image_masked,real_a_image_masked)# 生成画像とオリジナルの差
    loss_g = reconstruct_error

    loss_g.backward(retain_graph=True)
		#Optimizerの更新
    optimizerG.step()
	
    #####################################################################
    #Discriminatorを走らせる
    #####################################################################

    #center..中央の定数
    center = math.floor(image_size / 2)
    d = math.floor(Local_Window / 4) #trim(LocalDiscriminator用の窓)
    d2 = math.floor(Local_Window / 2) #L1Loss用の純粋な生成画像と同じサイズの窓用,所謂Mc

    #fake_b_imageはfake_b_image_rawにreal_a_imageを埋めたもの
    fake_b_image = real_a_image.clone()
    fake_b_image[:,:,center-d:center+d,center-d:center+d] = fake_b_image_raw[:,:,center-d:center+d,center-d:center+d]
    

    fake_c_image = torch.Tensor(opt.batchSize,1,Local_Window,Local_Window)
    real_c_image = torch.Tensor(opt.batchSize,1,Local_Window,Local_Window)
    #128*128の窓を作成 (あとで裏に回れるようにする)
    fake_c_image = fake_b_image[:,:,Mdpos_x-d2:Mdpos_x+d2,Mdpos_y-d2:Mdpos_y+d2]#↑でランダムに決めた位置
    real_c_image = real_a_image[:,:,Mdpos_x-d2:Mdpos_x+d2,Mdpos_y-d2:Mdpos_y+d2]#↑でランダムに決めた位置

    #GlobalDiscriminatorの起動

    #偽画像とマスクの結合..4次元に  
    fake_b_image_4d = torch.cat((fake_b_image,mask_channel_float),1)
    fake_c_image_4d = torch.cat((fake_c_image,white_channel_float_128),1)

    #真画像とマスクの結合..4次元に
    real_a_image_4d = torch.cat((real_a_image,random_mask_float_64),1)
    real_c_image_4d = torch.cat((real_c_image,white_channel_float_128),1) #LocalにもMdをかけるのか？←実質同じのため一旦All1のフィルタを入れる
    #pred_realは正しい画像を入力としたときの尤度テンソル(batchSize*1024*1*1)
    pred_realD_Global =  netD_Global.forward(real_a_image_4d)
    pred_realD_Local  =  netD_Local.forward(real_c_image_4d)
    pred_fakeD_Global = netD_Global.forward(fake_b_image_4d) #pred_falke=D(C(x,Mc),Mc)
    pred_fakeD_Local = netD_Local.forward(fake_c_image_4d) #pred_falke=D(C(x,Mc),Mc)

    pred_realD = torch.cat((pred_realD_Global,pred_realD_Local),1)
    pred_fakeD = torch.cat((pred_fakeD_Global,pred_fakeD_Local),1)

    #pred_fakeは偽生成画像を入力としたときの尤度テンソル
    pred_fakeD_Global = netD_Global.forward(fake_b_image_4d) #pred_falke=D(C(x,Mc),Mc)


    loss_d_realD = criterionBCE(pred_realD, true_label_tensor)
    loss_d_fakeD = criterionBCE(pred_fakeD, false_label_tensor) #ニセモノ-ホンモノをニセモノと判断させたいのでfalse
    #loss_d_realG_Local = criterionBCE(pred_realD_Local, true_label_tensor)
    #loss_d_fakeG_Local = criterionBCE(pred_fakeD_Local, false_label_tensor) #ニセモノ-ホンモノをニセモノと判断させたいのでfalse
		#
		

    #2つのロスの足し合わせ
    loss_d =  loss_d_realD + loss_d_fakeD 
    #backward
    loss_d.backward()
    

    #optimizerの更新
    optimizerD_Global.step()


    print("===> Epoch[{}]({}/{}):		Loss_G: {:.4f}  Loss_D_Global: {:.4f}".format(
        epoch, iteration, len(training_data_loader), loss_g.item(), loss_d.item()))
    if(iteration == 1):
      tensor_plot2image(fake_b_image_raw,'fakeC_Raw_Last_Epoch_{}'.format(epoch),iteration)
      tensor_plot2image(fake_b_image,'fakeC_Last_Epoch_{}'.format(epoch),iteration)


#12/11テストタスクを全部↑に引っ越す
def test(epoch):
  avg_psnr = 0
  with torch.no_grad():
    for batch in testing_data_loader: 
      #input, target = Variable(batch[0]), Variable(batch[1])
      if opt.cuda:
        input = input.cuda()
        target = target.cuda()

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
  net_dg_model_out_path = "checkpoint/{}/netDg_model_epoch_{}.pth".format(opt.dataset, epoch)
  torch.save(netD_Global, net_dg_model_out_path)
  print("Checkpoint saved to {}".format("checkpoint" + opt.dataset))


def checkpoint_total(epoch):
  if not os.path.exists("checkpoint"):
    os.mkdir("checkpoint")
  if not os.path.exists(os.path.join("checkpoint", opt.dataset)):
    os.mkdir(os.path.join("checkpoint", opt.dataset))
  net_g_model_out_path = "checkpoint/{}/Total_netG_model_epoch_{}.pth".format(opt.dataset, epoch)
  net_dg_model_out_path = "checkpoint/{}/Total_netDg_model_epoch_{}.pth".format(opt.dataset, epoch)
  torch.save(netG, net_g_model_out_path)
  torch.save(netD_Global, net_dg_model_out_path)
  print("Checkpoint saved to {}".format("checkpoint" + opt.dataset))

disc_only_epoch = 10
total_epoch = 50

for epoch in range(1, disc_only_epoch + 1):
#discriminatorのtrain
  train(epoch)
  checkpoint(epoch)


for epoch in range(1, total_epoch + 1):
#discriminatorのtrain
  total_train(epoch)
	
  checkpoint_total(epoch)








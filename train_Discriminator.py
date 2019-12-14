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
from   networks               import define_G, define_D, GANLoss, print_network, define_D_Edge,define_D_Global,define_D_Local
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
parser.add_argument('--G_model', type=str, default='checkpoint/testing_modelG_25.pth', help='model file to use')

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

#先ずGeneratorを読み込んで入れる
netG = torch.load(opt.G_model)
#NetDを3つ構築するのがよい
#netD = define_D(opt.input_nc + opt.output_nc, opt.ndf, 'batch', False, [0])
#そもそもいくつが入力なのか
disc_input_nc = 4
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
criterionCROSS = nn.CrossEntropyLoss()
criterionBCE = nn.BCELoss()
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

real_a = torch.FloatTensor(opt.batchSize, opt.input_nc, 256, 256)
real_b = torch.FloatTensor(opt.batchSize, opt.output_nc, 256, 256)

if opt.cuda:
  #netD = netD.cuda()
  netD_Global = netD_Global.cuda()
  netD_Local  = netD_Local.cuda()
  netD_Edge   = netD_Edge.cuda()

  netG = netG.cuda()
  criterionGAN = criterionGAN.cuda()
  criterionL1 = criterionL1.cuda()
  criterionMSE = criterionMSE.cuda()
  real_a = real_a.cuda()
  real_b = real_b.cuda()

real_a = Variable(real_a)
real_b = Variable(real_b)

#mask_channelは1*256*256の白黒マスク
center = math.floor(image_size / 2)
d = math.floor(Local_Window / 4) #4(窓サイズの1/4)
d2 = math.floor(Local_Window / 2) 

##白、黒チャネルの定義
black_channel_boolen = torch.full((opt.batchSize,1,image_size,image_size),False,dtype=bool)
white_channel_boolen = torch.full((opt.batchSize,1,hall_size,hall_size), True,dtype=bool)
black_channel_float = torch.full((opt.batchSize,1,image_size,image_size),False)
white_channel_float = torch.full((opt.batchSize,1,hall_size,hall_size), True)
##Mc(inputMask)の定義
mask_channel_float = black_channel_float.clone() #mask_channel=Mcに相当,Gが穴を開けた位置(必ず中央)
mask_channel_float[:,:,center - d:center+d,center - d:center+d] = white_channel_float
mask_channel_boolen = black_channel_boolen.clone()
mask_channel_boolen[:,:,center - d:center+d,center - d:center+d] = white_channel_boolen
##Md(RandomMask)の定義
random_mask_float = black_channel_float.clone() #random_channel=Mdに相当,毎iterationランダムな位置に穴を空ける
random_mask_boolen = black_channel_boolen.clone()

false_tensor = Variable(torch.LongTensor())
false_tensor  = torch.zeros(opt.batchSize,1024,1,1)
true_tensor = Variable(torch.LongTensor())
true_tensor  = torch.ones(opt.batchSize,1024,1,1)

seed = random.seed(1297)
padding = 16 #Mdが窓を生成し得る位置がが端から何ピクセル離れているか

if opt.cuda:
  mask_channel_boolen = mask_channel_boolen.cuda()
  mask_channel_float = mask_channel_float.cuda()
  random_mask_boolen = mask_channel_boolen.cuda()
  random_mask_float = mask_channel_float.cuda()
  true_tensor = true_tensor.cuda()
  false_tensor = false_tensor.cuda()
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
    #Md用の穴の場所決め
 
    #LocalDiscrimninator用のMd窓←LoaclDiscriminatorに窓は必要なくないか？
    #
    Mdpos_x = random.randint(0 + padding,image_size - hall_size - padding)
    Mdpos_y = random.randint(0 + padding,image_size - hall_size - padding)
    #
    random_mask_float[:,:,Mdpos_x:Mdpos_x+hall_size,Mdpos_y:Mdpos_y+hall_size] = white_channel_float
    random_mask_boolen[:,:,Mdpos_x:Mdpos_x+hall_size,Mdpos_y:Mdpos_y+hall_size] = white_channel_boolen


    real_a_cpu, real_b_cpu = batch[0], batch[1]#batchは元画像？
    
    real_a.data.resize_(real_a_cpu.size()).copy_(real_a_cpu)
    real_b.data.resize_(real_b_cpu.size()).copy_(real_b_cpu)
    #####################################################################
    #先ずGeneratorを起動して補完モデルを行う
    #####################################################################
    #fake_start_imageは単一画像中の平均画素でfillされている

    fake_start_image = torch.clone(real_a)
    for i in range(0, opt.batchSize):
      fake_start_image[i][0] = torch.mean(real_a[i][0])
      fake_start_image[i][1] = torch.mean(real_a[i][1])
      fake_start_image[i][2] = torch.mean(real_a[i][2])

    #fake_start_image2を穴のサイズにトリムしたもの
    fake_start_image2 = fake_start_image[:][:][0:hall_size][0:hall_size]
    fake_start_image2.resize_(opt.batchSize,opt.input_nc,hall_size,hall_size)

    #12/10real_cは真値の中央に平均画素を詰めたもの
    center = math.floor(image_size / 2)
    d = math.floor(Local_Window / 4) #4(窓サイズの1/4)
    real_c = real_b.clone()    
    real_c[:,:,center - d:center+d,center - d:center+d] = fake_start_image[:,:,center - d:center+d,center - d:center+d]

  
    real_c_4d = torch.cat((real_c,mask_channel_float),1)


    #2回目のジェネレータ起動(forwardをするため)
    fake_c_raw = netG(real_c_4d) # C(x,Mc)
    #fake_c = real_b.clone()#↓で穴以外はreal_bで上書きする
    #fake_c[:,:,center - d:center+d,center - d:center+d] = fake_c_raw[:,:,center - d:center+d,center - d:center+d]


    #####################################################################
    #Discriminatorを走らせる
    #####################################################################

    center = math.floor(image_size / 2)
    d = math.floor(Local_Window / 2) #trim(LocalDiscriminator用の窓)
    d2 = math.floor(Local_Window / 4) #L1Loss用の純粋な生成画像と同じサイズの窓用,所謂Mc

    #fake_cはfake_c_rawにreal_bを埋めたもの
    fake_c = real_b.clone()
    fake_c[:,:,center-d:center+d,center-d:center+d] = fake_c_raw[:,:,center-d:center+d,center-d:center+d]
        
    #GlobalDiscriminatorの起動

    #厳密にfake_c_rawが　C(x,Mc)なのか　fake_cが C(x,Mc)なのかが若干不明(fake_cな気もする)
    real_b_4d = torch.cat((real_b,random_mask_float),1)
    #pred_realは正しい画像を入力としたときの尤度テンソル(bat*3*256*256)
    pred_realG =  netD_Global.forward(real_b_4d)

    fake_c_4d = torch.cat((fake_c,mask_channel_float),1)

    #pred_fakeは偽生成画像を入力としたときの尤度テンソル(bat*3*256*256)
    pred_fakeG = netD_Global.forward(fake_c_4d) #pred_falke=D(C(x,Mc),Mc)

    

    #loss_d_realG = torch.log(criterionBCE(pred_realG,true_tensor))
    #loss_d_fakeG = torch.log(criterionBCE(pred_fakeG, false_tensor)) #ニセモノ-ホンモノをニセモノと判断させたいのでfalse

    #loss_d = loss_d_realG + loss_d_fakeG
    loss_d_realG = criterionBCE(pred_realG, true_tensor)
    loss_d_fakeG = criterionBCE(pred_fakeG, false_tensor) #ニセモノ-ホンモノをニセモノと判断させたいのでfalse

    #logを使っているとlossが0になると困る
    loss_d =  loss_d_realG + loss_d_fakeG 

    loss_d.backward()
    

 
    optimizerD_Global.step()


    print("===> Epoch[{}]({}/{}):  Loss_D_Global: {:.4f}".format(
       epoch, iteration, len(training_data_loader),  loss_d_fakeG.item()))
 
  

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
  #net_g_model_out_path = "checkpoint/{}/netG_model_epoch_{}.pth".format(opt.dataset, epoch)
  net_dg_model_out_path = "checkpoint/{}/netDg_model_epoch_{}.pth".format(opt.dataset, epoch)
  #net_dl_model_out_path = "checkpoint/{}/netDl_model_epoch_{}.pth".format(opt.dataset, epoch)
  #net_de_model_out_path = "checkpoint/{}/netDe_model_epoch_{}.pth".format(opt.dataset, epoch)
 # torch.save(netG, net_g_model_out_path)
  torch.save(netD_Global, net_dg_model_out_path)
#  torch.save(netD_Local, net_dl_model_out_path)
#  torch.save(netD_Edge, net_d_model_out_path)
  print("Checkpoint saved to {}".format("checkpoint" + opt.dataset))



 


for epoch in range(1, opt.nEpochs + 1):
  train(epoch)
  #test(epoch)
  
  checkpoint(epoch)

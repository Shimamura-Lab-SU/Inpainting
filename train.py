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

import random

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
parser.add_argument('--lr', type=float, default=0.00015, help='Learning Rate. Default=0.002')
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
netG = define_G(opt.input_nc, opt.output_nc, opt.ngf, 'batch', False, [0])
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
optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerD_Global = optim.Adam(netD_Global.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerD_Local = optim.Adam(netD_Local.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerD_Edge = optim.Adam(netD_Edge.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

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

def tensor_plot2image(__input,name,iteration=1):
  if(iteration == 1):
    path = os.getcwd() + '\\testing_output\\'
    vutils.save_image(__input.detach(), path + name + '.jpg')
    print('saved testing image')


def train(epoch):

  #batch_init = torch.tensor()
  for iteration, batch in enumerate(training_data_loader, 1):
    # forward
    #if (iteration == 0):
      #real_a_cpu, real_b_cpu = batch[0], batch[1]
      #batch_init = batch
    #else:
      #batch = batch_init
    real_a_cpu, real_b_cpu = batch[0], batch[1]#batchは元画像？

    
    real_a.data.resize_(real_a_cpu.size()).copy_(real_a_cpu)
    real_b.data.resize_(real_b_cpu.size()).copy_(real_b_cpu)

    #12/2 Md(ランダムノイズ)
    #random_d = torch.rand(real_a.shape, dtype=real_a.dtype)  
    #random_d.new_full(random_d.shape,45)
    fake_start_image = torch.clone(real_a)

    for i in range(0, opt.batchSize):
      fake_start_image[i][0] = torch.mean(real_a[i][0])
      fake_start_image[i][1] = torch.mean(real_a[i][1])
      fake_start_image[i][2] = torch.mean(real_a[i][2])

    center = math.floor(image_size / 2)
    d = math.floor(Local_Window / 4) #2→4(窓サイズの1/4)

    #tensor_plot2image(fake_start_image,'fakestart',iteration)
    real_c = real_b.clone() #参照渡しになっていたものを値渡しに変更
    
    real_c[:,:,center - d:center+d,center - d:center+d] = fake_start_image[:,:,center - d:center+d,center - d:center+d]

    #fake_cはreal_cをGeneratorにかけたもの
    tensor_plot2image(real_c,'realC',iteration)
    fake_c_raw = netG(real_b) #穴画像
    fake_c = real_b.clone()#↓で穴以外はreal_bで上書きする

    

    fake_c[:,:,center - d:center+d,center - d:center+d] = fake_c_raw[:,:,center - d:center+d,center - d:center+d]


    #fake_start_image2 = fake_start_image[:][:][0:hall_size][0:hall_size]
    #fake_start_image2.resize_(opt.batchSize,opt.input_nc,hall_size,hall_size)
    #fake_b_hallsize = netG(real_b) #fake_bが偽画像？←そうだよ

    

    #tensor_plot2image(real_a,'realA_0',iteration)
    #tensor_plot2image(real_b,'realB_0',iteration)
    #tensor_plot2image(fake_b_hallsize,'fakeB_0',iteration)
    #vutils.save_image(real_a[0][0].detach(), '{}\\real_sample00_{:03d}.png'.format(os.getcwd() + '\\testing_output', epoch,normalize=True, nrow=8))
     
  
    #fake_b = real_a
#fakebの穴以外の箇所をrealaで上書きする
    center = math.floor(image_size / 2)
    d = math.floor(Local_Window / 4) #2→4(窓サイズの1/4)

    fake_b = real_b.clone() #参照渡しになっていたものを値渡しに変更
    #fake_b[:,:,center - d:center+d,center - d:center+d] = copy(fake_b_hallsize)

    #fake_b = fake_b_hallsize

    ############################
    # (1) Update D network: maximize log(D(x,y)) + log(1 - D(x,G(x)))
    # ペアで学習しているので　本物vsDiscriminator 偽物 vs Discriminator で執り行うのはどっちも同じ
    ###########################
    #optimizerD_Global.zero_grad()
   # optimizerD_Local.zero_grad()
   # optimizerD_Edge.zero_grad()
    # train with fake

    #realAとFakebの穴周辺を切り出したものを用意する
    center = math.floor(image_size / 2)
    d = math.floor(Local_Window / 2) #trim(LocalDiscriminator用の窓)
    d2 = math.floor(Local_Window / 4) #L1Loss用の純粋な生成画像と同じサイズの窓用,所謂Mc

    real_a_trim = copy(real_a[:,:,center-d:center+d,center-d:center+d]) 
    real_b_trim = copy(real_b[:,:,center-d:center+d,center-d:center+d]) 
    fake_b_trim = copy(fake_b[:,:,center-d:center+d,center-d:center+d])

    real_a_trim2 = copy(real_a[:,:,center-d2:center+d2,center-d2:center+d2]) 
    real_b_trim2 = copy(real_b[:,:,center-d2:center+d2,center-d2:center+d2]) 
    fake_b_trim2 = copy(fake_b[:,:,center-d2:center+d2,center-d2:center+d2])


    fake_ab = torch.cat((real_b, fake_b), 1)
    fake_ab_trim = torch.cat((real_b_trim, fake_b_trim), 1)


    #tensor_plot2image(real_a,'realA_1',iteration)
    #tensor_plot2image(real_b,'realB_1',iteration)
    #tensor_plot2image(fake_b,'fakeB_1',iteration)

    #tensor_plot2image(real_a_trim,'realA_2',iteration)
    #tensor_plot2image(real_b_trim,'realB_2',iteration)
    #tensor_plot2image(fake_b_trim,'fakeB_2',iteration)
    #tensorを画像として扱うor .. トリミングする (後者を選択?)

    #reala,fakebのエッジ抽出
    #real_a_trim_8bit = (real_a_trim/256).astype('unit8') 

    #real_a_canny = cv2.Canny(real_a_trim, 100,600)
    #fake_b_canny = cv2.Canny(fake_b_trim, 100,600)
    #canny_ab = torch.cat((real_a_canny,fake_b_canny), 1)

    detatched_trim = fake_ab_trim.detach()
    detatched = fake_ab.detach()#detatched.shape ..[batchsize,6,256,256]
    #detatched_canny = canny_ab.detach()
    #グローバルDiscriminator


    #pred_fakeG = netD_Global.forward(fake_ab.detach()) #pred_dakeが偽画像を一時保存している
    #loss_d_fakeG = criterionGAN(pred_fakeG, False) #奥の引数はペアにした者が本か偽かを示すラベル

    #ローカルDiscriminator
    #pred_fakeL = netD_Local.forward(fake_ab_trim.detach()) #pred_dakeが偽画像を一時保存している
    #loss_d_fakeL = criterionGAN(pred_fakeL, False)

    #エッジDiscriminator(途中まで)
    #pred_fakeE = netD_Edge.forward(detatched_canny) #pred_dakeが偽画像を一時保存している
    #loss_d_fakeE = criterionGAN(pred_fakeE, False)

    # train with real
    #real_ab = torch.cat((real_b, real_b), 1) #torch.catはテンソルの連結splitの逆
    #real_ab_trim = torch.cat((real_b_trim, real_b_trim), 1)
    
    #pred_realG = netD_Global.forward(real_ab.detach())
    #loss_d_realG = criterionGAN(pred_realG, True)
    #pred_realL = netD_Local.forward(real_ab_trim.detach())
    #loss_d_realL = criterionGAN(pred_realL, True)
    #pred_realE = netD_Edge.forward(real_ab)
    #loss_d_realE = criterionGAN(pred_realE, True)
    # Combined loss
    #loss_d_fake = (loss_d_fakeG + loss_d_fakeL + loss_d_fakeE ) / 3 
    #loss_d_real = (loss_d_realG + loss_d_realL + loss_d_realE ) / 3 
    #loss_d_fake = (loss_d_fakeG + loss_d_fakeL) / 2
    #loss_d_real = (loss_d_realG + loss_d_realL) / 2 

    #loss_d = loss_d_fake + loss_d_real
    #loss_d.backward(retain_graph=True)
    #optimizerD_Global.step()
    #optimizerD_Local.step()
    #optimizerD_Edge.step()

    ############################
    # (2) Update G network: maximize log(D(x,G(x))) + L1(y,G(x))
    ##########################
   # optimizerG.zero_grad()
    # First, G(A) should fake the discriminator
    #fake_ab = torch.cat((real_a, fake_b), 1)
    #real_ab = torch.cat((real_b,real_b), 1)

    


   # pred_fakeG = netD_Global.forward(fake_ab.detach()) 
   # pred_fakeL = netD_Local.forward(fake_ab_trim.detach()) 
 
   # true_tensor = false_tensor = torch.clone(pred_fakeG)
   # true_tensor[:][:][:][:] = 1  #現状すべて0になっている
   # false_tensor[:][:][:][:] = 0


   # loss_g1 = criterionMSE(pred_fakeG, true_tensor)
   # loss_g2 = criterionMSE(pred_fakeL, true_tensor) #lossを統一することによってGeneratorが正しく機能するか？

    #12/9新しくforwardを導入。fakeb_hallsizeをもう一度作ってもらう
    center = math.floor(image_size / 2)
    d = math.floor(Local_Window / 4) 
    
    fake_c_raw = netG.forward(real_b)#穴画像
    fake_c = real_b.clone()#↓で穴以外はreal_bで上書きする
    fake_c[:,:,center - d:center+d,center - d:center+d] = fake_c_raw[:,:,center - d:center+d,center - d:center+d]



    #fake_b_hallsize = netG.forward(real_b)
    reconstruct_error = criterionMSE(fake_c_raw, real_b) # 生成画像とオリジナルの差
    #vutils.save_image(fake_c_raw.detach(), '{}\\real_sample00_{:03d}.png'.format(os.getcwd() + '\\testing_output', epoch,normalize=True, nrow=8))
    tensor_plot2image(fake_c_raw,'fakeb_{}'.format(epoch),iteration)
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
      vutils.save_image(fake_c_raw.detach(), '{}\\fake_samples_{:03d}.png'.format(os.getcwd() + '\\testing_output', epoch,normalize=True, nrow=8))

  #1epoch毎に出力してみる
  
def test(epoch):
  avg_psnr = 0
  with torch.no_grad():
    for batch in testing_data_loader: 
      input, target = Variable(batch[0]), Variable(batch[1])
      if opt.cuda:
        input = input.cuda()
        target = target.cuda()

      prediction = netG(input)
      mse = criterionMSE(prediction, target)
      psnr = 10 * log10(1 / mse.item())
      avg_psnr += psnr
    print("===> Avg. PSNR: {:.4f} dB".format(avg_psnr / len(testing_data_loader)))
     #チェックポイントの段階のモデルからアウトプットを作成する
  
    #if opt.cuda:
    #  netG = netG.cuda()
    #  input = input.cuda()

     # out = netG(input)
    #  out = out.cpu()
    #  out_img = out.data[0]  
    #  save_img(out_img, "checkpoint/{}/{}/{}".format(epoch,opt.dataset, image_name))


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
  test(epoch)
  
  checkpoint(epoch)

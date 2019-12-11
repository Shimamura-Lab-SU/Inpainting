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
training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True)
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
#criterionL1 = nn.L1Loss()
#criterionMSE = nn.MSELoss()

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
  #criterionL1 = criterionL1.cuda()
  #criterionMSE = criterionMSE.cuda()
  real_a = real_a.cuda()
  real_b = real_b.cuda()

real_a = Variable(real_a)
real_b = Variable(real_b)

def train(epoch):

  for iteration, batch in enumerate(training_data_loader, 1):
    # forward
    real_a_cpu, real_b_cpu = batch[0], batch[1]#batchは元画像？
    real_a.data.resize_(real_a_cpu.size()).copy_(real_a_cpu)
    real_b.data.resize_(real_b_cpu.size()).copy_(real_b_cpu)
    fake_b_all = netG(real_a) #fake_bが偽画像？←そうだよ
    #fake_b = real_a
#fakebの穴以外の箇所をrealaで上書きする
    center = math.floor(image_size / 2)
    d = math.floor(Local_Window / 2)
    fake_b = copy(real_a)
    fake_b[:,:,center - d:center+d,center - d:center+d] = copy(fake_b_all[:,:,center - d:center+d,center - d:center+d])

    ############################
    # (1) Update D network: maximize log(D(x,y)) + log(1 - D(x,G(x)))
    # ペアで学習しているので　本物vsDiscriminator 偽物 vs Discriminator で執り行うのはどっちも同じ
    ###########################
    optimizerD_Global.zero_grad()
    optimizerD_Local.zero_grad()
    optimizerD_Edge.zero_grad()
    # train with fake

    #realAとFakebの穴周辺を切り出したものを用意する
    center = math.floor(image_size / 2)
    d = math.floor(Local_Window / 2)

    real_a_trim = copy(real_a[:,:,center-d:center+d,center-d:center+d]) 
    fake_b_trim = copy(fake_b[:,:,center-d:center+d,center-d:center+d])


    fake_ab = torch.cat((real_a, fake_b), 1)
    fake_ab_trim = torch.cat((real_a_trim, fake_b_trim), 1)
    #tensorを画像として扱うor .. トリミングする (後者を選択?)

    #reala,fakebのエッジ抽出
    real_a_trim_8bit = (real_a_trim/256).astype('unit8') 

    real_a_canny = cv2.Canny(real_a_trim, 100,600)
    fake_b_canny = cv2.Canny(fake_b_trim, 100,600)
    canny_ab = torch.cat((real_a_canny,fake_b_canny), 1)

    detatched_trim = fake_ab_trim.detach()
    detatched = fake_ab.detach()#detatched.shape ..[batchsize,6,256,256]
    detatched_canny = canny_ab.detach()
    #グローバルDiscriminator

    pred_fakeG = netD_Global.forward(detatched) #pred_dakeが偽画像を一時保存している
    loss_d_fakeG = criterionGAN(pred_fakeG, False)

    #ローカルDiscriminator
    pred_fakeL = netD_Local.forward(detatched_trim) #pred_dakeが偽画像を一時保存している
    loss_d_fakeL = criterionGAN(pred_fakeL, False)

    #エッジDiscriminator(途中まで)
    pred_fakeE = netD_Edge.forward(detatched_canny) #pred_dakeが偽画像を一時保存している
    loss_d_fakeE = criterionGAN(pred_fakeE, False)

    # train with real
    real_ab = torch.cat((real_a, real_b), 1) #torch.catはテンソルの連結splitの逆
    pred_realG = netD_Global.forward(real_ab)
    loss_d_realG = criterionGAN(pred_realG, True)
    pred_realL = netD_Local.forward(real_ab)
    loss_d_realL = criterionGAN(pred_realL, True)
    pred_realE = netD_Edge.forward(real_ab)
    loss_d_realE = criterionGAN(pred_realE, True)
    # Combined loss
    loss_d_fake = (loss_d_fakeG + loss_d_fakeL + loss_d_fakeE ) / 3 
    loss_d_real = (loss_d_realG + loss_d_realL + loss_d_realE ) / 3 

    loss_d = (loss_d_fake + loss_d_real  ) * 0.5
    loss_d.backward()
    optimizerD_Global.step()
    optimizerD_Local.step()
    optimizerD_Edge.step()



    ############################
    # (2) Update G network: maximize log(D(x,G(x))) + L1(y,G(x))
    ##########################
    optimizerG.zero_grad()
    # First, G(A) should fake the discriminator
    fake_ab = torch.cat((real_a, fake_b), 1)
    

    pred_fakeG = netD_Global.forward(fake_ab) 
    pred_fakeL = netD_Local.forward(fake_ab) 
    pred_fakeE = netD_Edge.forward(fake_ab) 
    loss_g1 = criterionGAN(pred_fakeG, True)
    loss_g2 = criterionGAN(pred_fakeL, True)
    loss_g3 = criterionGAN(pred_fakeE, True)
     # Second, G(A) = B
    #loss_g_l1 = criterionL1(fake_b, real_b) * opt.lamb
    #loss_g.backward()
    loss_g1.backward()
    loss_g2.backward()
    loss_g3.backward()
    optimizerG.step()
    print("===> Epoch[{}]({}/{}): Loss_D: {:.4f} Loss_G: {:.4f}".format(
        epoch, iteration, len(training_data_loader), loss_d.item(), loss_g1.item()))
    if(iteration == len(training_data_loader)):
      vutils.save_image(fake_b.detach(), '{}\\fake_samples_{:03d}.png'.format(os.getcwd() + '\\checkpoint_output', epoch,normalize=True, nrow=8))

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
  net_d_model_out_path = "checkpoint/{}/netD_model_epoch_{}.pth".format(opt.dataset, epoch)
  torch.save(netG, net_g_model_out_path)
  torch.save(netD, net_d_model_out_path)
  print("Checkpoint saved to {}".format("checkpoint" + opt.dataset))



 


for epoch in range(1, opt.nEpochs + 1):
  train(epoch)
  test(epoch)
  
  checkpoint(epoch)

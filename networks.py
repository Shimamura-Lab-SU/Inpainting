import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import math

import torchvision.utils as vutils
import torch.nn.functional as F

from pytorch_memlab import profile

torch.device
#ソーベル法に変更
def edge_detection(__input,is_gpu = True):
    kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)  # convolution filter
    kernel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], np.float32)  # convolution filter

        # [out_ch, in_ch, .., ..] : channel wiseに計算
    edge_k_x = torch.as_tensor(kernel_x.reshape(1, 1, 3, 3)) #cudaでやる場合デバイスを指定する
    edge_k_y = torch.as_tensor(kernel_y.reshape(1, 1, 3, 3)) #cudaでやる場合デバイスを指定する
    if(is_gpu):
      edge_k_x = edge_k_x.cuda()
      edge_k_y = edge_k_y.cuda()

        # エッジ検出はグレースケール化してからやる
    color = __input  # color image [1, 3, H, W]
    gray_kernel = np.array([0.299, 0.587, 0.114], np.float32).reshape(1, 3, 1, 1)  # color -> gray kernel
    gray_k = torch.as_tensor(gray_kernel)

    if(is_gpu):
      gray_k = gray_k.cuda()
    gray = torch.sum(color * gray_k, dim=1, keepdim=True)  # grayscale image [1, 1, H, W]

        # エッジ検出
    edge_image_x = F.conv2d(gray, edge_k_x, padding=1)
    edge_image = F.conv2d(edge_image_x, edge_k_y, padding=1)

    return edge_image

def weights_init(m):
  classname = m.__class__.__name__
  if classname.find('Conv') != -1:
    m.weight.data.normal_(0.0, 0.02)
  elif classname.find('BatchNorm2d') != -1 or classname.find('InstanceNorm2d') != -1:
    m.weight.data.normal_(1.0, 0.02)
    m.bias.data.fill_(0)


def get_norm_layer(norm_type):
  if norm_type == 'batch':
    norm_layer = nn.BatchNorm2d
  elif norm_type == 'instance':
    norm_layer = nn.InstanceNorm2d
  else:
    print('normalization layer [%s] is not found' % norm_type)
  return norm_layer


def define_G(input_nc, output_nc, ngf, norm='batch', use_dropout=False, gpu_ids=[0]):
  norm_layer = get_norm_layer(norm_type=norm)
  netG = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9, gpu_ids=gpu_ids)
  netG.apply(weights_init)
  return netG


#いる #いる　#いる #いらない #いる
def define_D_Global(input_nc,output_nc, ndf,  gpu_ids=[0]):
  netD_Global = Global_Discriminator(input_nc, output_nc,  ndf,gpu_ids=gpu_ids)
  netD_Global.apply(weights_init)
  return netD_Global

def define_D_Local(input_nc, output_nc,ndf, gpu_ids=[0]):
  netD_Local = Local_Discriminator(input_nc, output_nc, ndf, gpu_ids=gpu_ids)
  netD_Local.apply(weights_init)
  return netD_Local

def define_D_Edge(input_nc, output_nc, ndf, gpu_ids=[0]):
  netD_Edge = Edge_Discriminator(input_nc, output_nc, ndf,  gpu_ids=gpu_ids)
  netD_Edge.apply(weights_init)
  return netD_Edge

def define_Concat(input_nc, output_nc, gpu_ids=[0]):
  net_Concat = Concatenation(input_nc, output_nc, gpu_ids)
  net_Concat.apply(weights_init)
  return net_Concat

def print_network(net):
  num_params = 0
  for param in net.parameters():
    num_params += param.numel()
  print(net)
  print('Total number of parameters: %d' % num_params)


# Defines the GAN loss which uses either LSGAN or the regular GAN.
class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                self.real_label_var = Variable(real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = Variable(fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor.cuda())


# Defines the generator that consists of Resnet blocks between a few
# downsampling/upsampling operations.
class ResnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, gpu_ids=[]):
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.gpu_ids = gpu_ids

        #入力層

        #モデルに付与する引数について kernel_sizeを設定した倍paddingは其れを半分にして切り捨てた値を与えるべし(paddingは入力の端っこを幾つ拡張するか)
        #conv1
        model = [nn.Conv2d(input_nc, ngf, kernel_size=5, stride=1, padding=2,dilation=1),
                 norm_layer(ngf, affine=True),
                 nn.ReLU(True)]

        #batch normalization のnormはuotputの次元数と同じ

        #[]で囲って配列でまとめてるだけなので不要なnormalizationは取り除く
        #conv2
        model += [nn.Conv2d(ngf * 1, ngf * 2,kernel_size = 3, stride = 2,padding = 1), norm_layer(ngf * 2, affine = True), nn.ReLU(True)]
        #conv3
        model += [nn.Conv2d(ngf * 2, ngf * 2,kernel_size = 3, stride = 1,padding = 1), norm_layer(ngf * 2, affine = True), nn.ReLU(True)]
        #----------------------
        #conv4
        model += [nn.Conv2d(ngf * 2, ngf * 4,kernel_size = 3, stride = 2,padding = 1), norm_layer(ngf * 4, affine = True), nn.ReLU(True)]
        #conv5
        model += [nn.Conv2d(ngf * 4, ngf * 4,kernel_size = 3, stride = 1,padding = 1), norm_layer(ngf * 4, affine = True), nn.ReLU(True)]
        #conv6
        model += [nn.Conv2d(ngf * 4, ngf * 4,kernel_size = 3, stride = 1,padding = 1), norm_layer(ngf * 4, affine = True), nn.ReLU(True)]
        #----------------------
        #Dilated conv7 dilationするとテンソルの次元数が下がって計算が合わなくなる 2 4 8　16 → 1 1 1 1
        model += [nn.Conv2d(ngf * 4, ngf * 4,kernel_size = 3, stride = 1,padding = 2, dilation = 2), norm_layer(ngf * 4, affine = True), nn.ReLU(True)]
        #Dilated conv8
        model += [nn.Conv2d(ngf * 4, ngf * 4,kernel_size = 3, stride = 1,padding = 4, dilation = 4), norm_layer(ngf * 4, affine = True), nn.ReLU(True)]
        #Dilated conv9
        model += [nn.Conv2d(ngf * 4, ngf * 4,kernel_size = 3, stride = 1,padding = 8, dilation = 8), norm_layer(ngf * 4, affine = True), nn.ReLU(True)]
        #Dilated conv10
        model += [nn.Conv2d(ngf * 4, ngf * 4,kernel_size = 3, stride = 1,padding = 16, dilation = 16), norm_layer(ngf * 4, affine = True), nn.ReLU(True)]
        #----------------------
        #conv11
        model += [nn.Conv2d(ngf * 4, ngf * 4,kernel_size = 3, stride = 1,padding = 1), norm_layer(ngf * 4, affine = True), nn.ReLU(True)]
        #conv12
        model += [nn.Conv2d(ngf * 4, ngf * 4,kernel_size = 3, stride = 1,padding = 1), norm_layer(ngf * 4, affine = True), nn.ReLU(True)]
        #deconv
        model += [nn.ConvTranspose2d(ngf * 4, ngf * 2,kernel_size = 4, stride = 2,padding = 1), norm_layer(ngf * 2, affine = True), nn.ReLU(True)]
        #conv
        model += [nn.Conv2d(ngf * 2, ngf * 2,kernel_size = 3, stride = 1,padding = 1), norm_layer(ngf * 2, affine = True), nn.ReLU(True)]
        #deconv  
        model += [nn.ConvTranspose2d(ngf * 2, ngf * 1,kernel_size = 4, stride = 2,padding = 1), norm_layer(ngf, affine = True), nn.ReLU(True)]
        #conv
        model += [nn.Conv2d(ngf * 1, int(ngf / 2),kernel_size = 3, stride = 1,padding = 1), norm_layer(int(ngf/ 2 ), affine = True), nn.ReLU(True)]
        #output
        model += [nn.Conv2d(int(ngf / 2) , output_nc,kernel_size = 3, stride = 1,padding = 1)]
        #model += [nn.Sigmoid()]

        self.model = nn.Sequential(*model)

    
    def forward(self, input):
      tensor_b = input.cuda()
      if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
        tensor_b = nn.parallel.data_parallel(self.model, tensor_b, self.gpu_ids)
      else:
        tensor_b = self.model(tensor_b)
      
      tensor_b = tensor_b.cpu()
      return tensor_b
      
    
    def forwardWithMasking(self, input, mask_size, batch_num = 1, image_size = 256):
      center = math.floor(image_size / 2)
      #from train_all import opt, center
      d = math.floor(mask_size/2)
      fake_start_image = torch.clone(input) #cp ←cpu

      for i in range(0, batch_num):#中心中心
        fake_start_image[i][0] = torch.mean(input[i][0])
        fake_start_image[i][1] = torch.mean(input[i][1])
        fake_start_image[i][2] = torch.mean(input[i][2])

      #fake_start_image2を穴のサイズにトリムしたもの
      #fake_start_image2 = fake_start_image[:][:][0:mask_size][0:mask_size]
      #fake_start_image2.resize_(opt.batchSize,opt.input_nc,mask_size,mask_size)

      tensor_b = input.clone()    
      tensor_b[:,:,center - d:center+d,center - d:center+d] = fake_start_image[:,:,center - d:center+d,center - d:center+d]

      return(self.forward(tensor_b))






#GlobalDiscriminator
class Global_Discriminator(nn.Module):
  def __init__(self, input_nc, output_nc, ndf= 64, gpu_ids=[] ):
      super(Global_Discriminator, self).__init__()
      self.gpu_ids = gpu_ids
      self.input_nc = input_nc
      self.output_nc = output_nc
      self.ndf =ndf
      #1

      model_conv = [nn.Conv2d(input_nc, ndf, kernel_size=5, stride=2, padding=2,dilation=1),nn.ReLU(True)]
      #conv2
      model_conv += [nn.Conv2d(ndf, ndf * 2, kernel_size=5, stride=2, padding=2,dilation=1), nn.ReLU(True)]
      #conv2
      model_conv += [nn.Conv2d(ndf * 2, ndf * 4, kernel_size=5, stride=2, padding=2,dilation=1), nn.ReLU(True)]
      model_conv += [nn.Conv2d(ndf * 4, ndf * 8, kernel_size=5, stride=2, padding=2,dilation=1), nn.ReLU(True)]

      model_conv += [nn.Conv2d(ndf * 8, ndf * 8, kernel_size=5, stride=2, padding=2,dilation=1), nn.ReLU(True)]
      model_conv += [nn.Conv2d(ndf * 8, ndf * 8, kernel_size=5, stride=2, padding=2,dilation=1), nn.ReLU(True)]

      model_dence = [nn.Linear(ndf * 8 * 4 * 4, output_nc)]
      model_dence += [nn.Sigmoid()] #sigmoidを入れるとBCELOSSを通れるようになるため

      self.model_conv = nn.Sequential(*model_conv)
      self.model_dence =  nn.Sequential(*model_dence)

  def forward(self, input):
    out = input.cuda()
    if self.gpu_ids and isinstance(out.data, torch.cuda.FloatTensor):
      #out = input.cuda()
      out = nn.parallel.data_parallel(self.model_conv, out, self.gpu_ids)
      #Viewで中間層から形状を変える
      #サイズを知りたい
      #size = out.size()

      out = out.view(out.size(0),-1) 
      out = nn.parallel.data_parallel(self.model_dence, out, self.gpu_ids) # 全結合層
      out = out.cpu()
      return out
    else:
      #out = input.cuda()
      out = self.model_conv(out)
      #Flatten
      out = out.view(out.size(0),-1)
      out = self.model_dence(out) 
      out = out.cpu()
      return out

  #Fake_RawにFakeをかぶせる前処理をしてからネットを走らせる場合
  def forwardWithCover(self, input,_input_real = torch.empty((1,1)), hole_size = 0, image_size = 256):
    #カバーを行う
    #from train_all import center,d
    center = math.floor(image_size / 2) #たいてい128
    d =  math.floor(hole_size / 2) #だいたい32    
    #fake_b_imageはfake_b_image_rawにreal_a_imageを埋めたもの
    tensor_b = _input_real.clone()
    tensor_b[:,:,center-d:center+d,center-d:center+d] = input[:,:,center-d:center+d,center-d:center+d]
    tensor_b = self.forward(tensor_b)
    return tensor_b

  def forwardWithTrim(self, input, _xpos = 0, _ypos = 0, trim_size = 0,batchSize = -1):
    #トリムを行う
    #from train_all import opt
    d = math.floor(trim_size / 2)
    tensor_a = torch.Tensor(opt.batchSize,1,trim_size,trim_size)
    tensor_a = input[:,:,_xpos-d:_xpos+d,_ypos-d:_ypos+d]

    return self.forward(tensor_a)

  def forwardWithTrimCover(self, input, _xpos = 0, _ypos = 0, trim_size = 0,_input_real = torch.empty((1,1)), hole_size = 0, image_size = 256, batchSize = -1):
    #カバーを行ったのちにトリムを行う
    #カバーを行う
    #from train_all import center,d,opt
    #fake_b_imageはfake_b_image_rawにreal_a_imageを埋めたもの
    center = math.floor(image_size / 2)
    d = math.floor(hole_size / 2)

    tensor_b = _input_real.clone()
    tensor_b[:,:,center-d:center+d,center-d:center+d] = input[:,:,center-d:center+d,center-d:center+d]
    #tensorbをinputとしてトリムを行う
    d2 = math.floor(trim_size / 2)
    tensor_a = torch.Tensor(batchSize,1,trim_size,trim_size)
    tensor_a = tensor_b[:,:,_xpos-d2:_xpos+d2,_ypos-d2:_ypos+d2]

    return self.forward(tensor_a)

#Concatenation
class Concatenation(nn.Module):
  def __init__(self, input_nc, output_nc, gpu_ids=[] ):
      super(Concatenation, self).__init__()
      self.gpu_ids = gpu_ids
      self.input_nc = input_nc
      self.output_nc = output_nc

      model = [nn.Linear(input_nc,output_nc)]

      model += [nn.Sigmoid()]
      self.model = nn.Sequential(*model)

  def forward(self, _global_input, _local_input):
    #catで入寮同士をつなぐ
    input = torch.cat((_global_input,_local_input),1)
    input = input.view(-1,2048)
    input = input.cuda()
    if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
        input = nn.parallel.data_parallel(self.model, input, self.gpu_ids)
    else:
        input = self.model(input)
    return input.cpu()
  
  def forward1(self, input):
    #catで入寮同士をつなぐ
    #input = torch.cat((_global_input,_local_input,_edge_input),1)
    #input = input.view(-1,3072)
    input = input.cuda()
    if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
        input = nn.parallel.data_parallel(self.model, input, self.gpu_ids)
    else:
        input = self.model(input)
    return input.cpu()


#LocalDiscriminator
class Local_Discriminator(nn.Module):

  def __init__(self, input_nc, output_nc ,ndf= 64,gpu_ids=[] ):
      super(Local_Discriminator, self).__init__()
      self.gpu_ids = gpu_ids
      self.input_nc = input_nc
      self.output_nc = output_nc
      self.ndf = ndf
      #モデルを畳み込み層(conv)と全結合層(dence)に分ける
      #1
      model_conv = [nn.Conv2d(input_nc, ndf, kernel_size=5, stride=2, padding=2,dilation=1),nn.ReLU(True)] #4
      #conv2
      model_conv += [nn.Conv2d(ndf, ndf * 2, kernel_size=5, stride=2, padding=2,dilation=1),nn.ReLU(True)] 
      #conv2
      model_conv += [nn.Conv2d(ndf * 2, ndf * 4, kernel_size=5, stride=2, padding=2,dilation=1),nn.ReLU(True)]
      model_conv += [nn.Conv2d(ndf * 4, ndf * 8, kernel_size=5, stride=2, padding=2,dilation=1),nn.ReLU(True)]

      model_conv += [nn.Conv2d(ndf * 8, ndf * 8, kernel_size=5, stride=2, padding=2,dilation=1),nn.ReLU(True)] #[b,512,4,4]がconv層の出力

      #FullConvolution層
      #model += [nn.Conv2d(ndf * 8, output_nc, 4, 1)]
      model_dence = [nn.Linear(ndf * 8 * 4 * 4 , output_nc)]
      model_dence += [nn.Sigmoid()]
      self.model_conv = nn.Sequential(*model_conv)
      self.model_dence =  nn.Sequential(*model_dence)
  @profile
  def forward(self, input):

    out = input.cuda()
    if self.gpu_ids and isinstance(out.data, torch.cuda.FloatTensor):
      out = nn.parallel.data_parallel(self.model_conv, out, self.gpu_ids)
      #Viewで中間層から形状を変える
      out = out.view(out.size(0),-1) 
      out = nn.parallel.data_parallel(self.model_dence, out, self.gpu_ids) # 全結合層
      out = out.cpu()
      return out      
    else:
      out = self.model_conv(out)
      #Flatten
      out = out.view(out.size(0),-1)
      out = self.model_dence(out) 
      out = out.cpu()
      return out

  def check_cnn_size(self, size_check ):
    out = self.model_conv(size_check)
    return out

  #Fake_RawにFakeをかぶせる前処理をしてからネットを走らせる場合
  def forwardWithCover(self, input,_input_real = torch.empty((1,1)), hole_size = 0,image_size = 256):
    #カバーを行う
    #from train_all import center,d
    #fake_b_imageはfake_b_image_rawにreal_a_imageを埋めたもの
    center = math.floor(image_size / 2)
    d = math.floor(hole_size / 2)
    tensor_b = _input_real.clone()
    tensor_b[:,:,center-d:center+d,center-d:center+d] = input[:,:,center-d:center+d,center-d:center+d]

    return(self.forward(tensor_b))

  def forwardWithTrim(self, input, _xpos = 0, _ypos = 0, trim_size = 0,batchSize = 1):
    #トリムを行う
    #from train_all import opt
    d = math.floor(trim_size / 2)
    tensor_a = torch.Tensor(batchSize,1,trim_size,trim_size)
    tensor_a = input[:,:,_xpos-d:_xpos+d,_ypos-d:_ypos+d]

    return(self.forward(tensor_a))



  def forwardWithTrimCover(self, input, _xpos = 0, _ypos = 0, trim_size = 0,_input_real = torch.empty((1,1)), hole_size = 0, image_size = 256, batchSize = -1):
    #カバーを行ったのちにトリムを行う
    #カバーを行う
    #from train_all import center,d,opt
    #fake_b_imageはfake_b_image_rawにreal_a_imageを埋めたもの
    center = math.floor(image_size / 2)
    d = math.floor(hole_size / 2)

    tensor_b = _input_real.clone()
    tensor_b[:,:,center-d:center+d,center-d:center+d] = input[:,:,center-d:center+d,center-d:center+d]
    #tensorbをinputとしてトリムを行う
    d2 = math.floor(trim_size / 2)
    tensor_a = torch.Tensor(batchSize,1,trim_size,trim_size)
    tensor_a = tensor_b[:,:,_xpos-d2:_xpos+d2,_ypos-d2:_ypos+d2]

    return(self.forward(tensor_a))


#EdgeDiscriminator
class Edge_Discriminator(nn.Module):
  def __init__(self, input_nc, output_nc ,ndf= 64,gpu_ids=[] ):
      super(Edge_Discriminator, self).__init__()
      self.gpu_ids = gpu_ids
      self.input_nc = 2 #決め打ち
      self.output_nc = output_nc
      self.ndf = ndf

      #1
      model_conv = [nn.Conv2d(input_nc, ndf, kernel_size=5, stride=2, padding=2,dilation=1),nn.ReLU(True)]
      #conv2
      model_conv += [nn.Conv2d(ndf, ndf * 2, kernel_size=5, stride=2, padding=2,dilation=1)]
      #conv2
      model_conv += [nn.Conv2d(ndf * 2, ndf * 4, kernel_size=5, stride=2, padding=2,dilation=1)]
      model_conv += [nn.Conv2d(ndf * 4, ndf * 8, kernel_size=5, stride=2, padding=2,dilation=1)]

      model_conv += [nn.Conv2d(ndf * 8, ndf * 8, kernel_size=5, stride=2, padding=2,dilation=1)]

      #1024次元にしたい
      #FullConvolution層  
      model_dence = [nn.Linear(ndf * 8 * 4 * 4 , output_nc)]
      model_dence += [nn.Sigmoid()]

      self.model_conv = nn.Sequential(*model_conv)
      self.model_dence = nn.Sequential(*model_dence)



  def forward(self, input):
    #入ってきたTensorをエッジ変換する
    input_edge = edge_detection(input[:,0:3,:,:],is_gpu=False)
    #2dはGray+Maskの2channel
    input_mask = input[:,3:4,:,:]
    input_2d = torch.cat((input_edge,input_mask),1)
    input_2d = input_2d.cuda()

    if self.gpu_ids and isinstance(input_2d.data, torch.cuda.FloatTensor):
      out = nn.parallel.data_parallel(self.model_conv, input_2d, self.gpu_ids)
      #Viewで中間層から形状を変える
      out = out.view(out.size(0),-1) 
      out = nn.parallel.data_parallel(self.model_dence, out, self.gpu_ids) # 全結合層
      out = out.cpu()
      return out      
    else:
      out = self.model_conv(input_2d)
      #Flatten
      out = out.view(out.size(0),-1)
      out = self.model_dence(out) 
      out = out.cpu()
      return out

  #Fake_RawにFakeをかぶせる前処理をしてからネットを走らせる場合
  def forwardWithCover(self, input,_input_real = torch.empty((1,1)), hole_size = 0, image_size = 256):
    #カバーを行う
    #from train_all import center,d
    #fake_b_imageはfake_b_image_rawにreal_a_imageを埋めたもの
    center = math.floor(image_size / 2)
    d = math.floor(hole_size / 2)


    tensor_b = _input_real.clone()
    tensor_b[:,:,center-d:center+d,center-d:center+d] = input[:,:,center-d:center+d,center-d:center+d]

    return(self.forward(tensor_b))

  def forwardWithTrim(self, input, _xpos = 0, _ypos = 0, trim_size = 0, batchSize = -1):
    #トリムを行う
    #from train_all import opt
    d = math.floor(trim_size / 2)
    tensor_a = torch.Tensor(batchSize,1,trim_size,trim_size)
    tensor_a = input[:,:,_xpos-d:_xpos+d,_ypos-d:_ypos+d]

    return(self.forward(tensor_a))

  def forwardWithTrimCover(self, input, _xpos = 0, _ypos = 0, trim_size = 0,_input_real = torch.empty((1,1)), hole_size = 0, image_size = 256, batchSize = -1):
    #カバーを行ったのちにトリムを行う
    #カバーを行う
    #from train_all import center,d,opt
    #fake_b_imageはfake_b_image_rawにreal_a_imageを埋めたもの
    center = math.floor(image_size / 2)
    d = math.floor(hole_size / 2)


    tensor_b = _input_real.clone()
    tensor_b[:,:,center-d:center+d,center-d:center+d] = input[:,:,center-d:center+d,center-d:center+d]
    #tensorbをinputとしてトリムを行う
    d2 = math.floor(trim_size / 2)
    tensor_a = torch.Tensor(batchSize,1,trim_size,trim_size)
    tensor_a = tensor_b[:,:,_xpos-d2:_xpos+d2,_ypos-d2:_ypos+d2]

    return(self.forward(tensor_a))





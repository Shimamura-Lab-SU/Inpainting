import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


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


def define_G(input_nc, output_nc, ngf, norm='batch', use_dropout=False, gpu_ids=[]):
  netG = None
  use_gpu = len(gpu_ids) > 0
  norm_layer = get_norm_layer(norm_type=norm)

  if use_gpu:
    assert(torch.cuda.is_available())

  netG = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9, gpu_ids=gpu_ids)

  if len(gpu_ids) > 0:
    netG.cuda()
  netG.apply(weights_init)
  return netG


def define_D(input_nc, ndf, norm='batch', use_sigmoid=False, gpu_ids=[]):
  netD = None
  use_gpu = len(gpu_ids) > 0
  norm_layer = get_norm_layer(norm_type=norm)

  if use_gpu:
    assert(torch.cuda.is_available())

  netD = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer, use_sigmoid=use_sigmoid, gpu_ids=gpu_ids)

  if use_gpu:
    netD.cuda()
  netD.apply(weights_init)
  return netD
#netD_Global = define_D_Global(hogehoge)
#netD_Local  = define_D_Local(hogehoge)
#netD_Edge   = define_D_Edge(hogehoge)

#いる #いる　#いる #いらない #いる
def define_D_Global(input_nc,output_nc, ndf,  gpu_ids=[]):
  netD_Global = None
  use_gpu = len(gpu_ids) > 0

  if use_gpu:
    assert(torch.cuda.is_available())

  netD_Global = Global_Discriminator(input_nc, output_nc,  ndf,gpu_ids=gpu_ids)

  if use_gpu:
    netD_Global.cuda()
  netD_Global.apply(weights_init)
  return netD_Global

def define_D_Local(input_nc, output_nc,ndf, gpu_ids=[]):
  netD_Local = None
  use_gpu = len(gpu_ids) > 0
  if use_gpu:
    assert(torch.cuda.is_available())

  netD_Local = Local_Discriminator(input_nc, output_nc, ndf, gpu_ids=gpu_ids)

  if use_gpu:
    netD_Local.cuda()
  netD_Local.apply(weights_init)
  return netD_Local

def define_D_Edge(input_nc, output_nc, ndf, gpu_ids=[]):
  netD_Edge = None
  use_gpu = len(gpu_ids) > 0

  if use_gpu:
    assert(torch.cuda.is_available())

  netD_Edge = Edge_Discriminator(input_nc, output_nc, ndf,  gpu_ids=gpu_ids)

  if use_gpu:
    netD_Edge.cuda()
  netD_Edge.apply(weights_init)
  return netD_Edge
'''
def define_Detector(input_nc,output_nc,gpu_ids=[],tensor=torch.FloatTensor):
  net_Detector = None
  use_gpu = len(gpu_ids) > 0

  if use_gpu:
    assert(torch.cuda.is_available())

  #net_Detector = Edge_Detector(input_nc,output_nc,gpu_ids=gpu_ids,tensor=tensor)

  if use_gpu:
    #net_Detector.cuda()
  net_Detector.apply(weights_init)
  return net_Detector 
'''
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
        model += [nn.Conv2d(ngf * 1, ngf * 2,kernel_size = 3, stride = 2,padding = 1), norm_layer(ngf * 2, affine = True)]
        #conv3
        model += [nn.Conv2d(ngf * 2, ngf * 2,kernel_size = 3, stride = 1,padding = 1), norm_layer(ngf * 2, affine = True)]
        #----------------------
        #conv4
        model += [nn.Conv2d(ngf * 2, ngf * 4,kernel_size = 3, stride = 2,padding = 1), norm_layer(ngf * 4, affine = True)]
        #conv5
        model += [nn.Conv2d(ngf * 4, ngf * 4,kernel_size = 3, stride = 1,padding = 1), norm_layer(ngf * 4, affine = True)]
        #conv6
        model += [nn.Conv2d(ngf * 4, ngf * 4,kernel_size = 3, stride = 1,padding = 1), norm_layer(ngf * 4, affine = True)]
        #----------------------
        #Dilated conv7 dilationするとテンソルの次元数が下がって計算が合わなくなる 2 4 8　16 → 1 1 1 1
        model += [nn.Conv2d(ngf * 4, ngf * 4,kernel_size = 3, stride = 1,padding = 2, dilation = 2), norm_layer(ngf * 4, affine = True)]
        #Dilated conv8
        model += [nn.Conv2d(ngf * 4, ngf * 4,kernel_size = 3, stride = 1,padding = 4, dilation = 4), norm_layer(ngf * 4, affine = True)]
        #Dilated conv9
        model += [nn.Conv2d(ngf * 4, ngf * 4,kernel_size = 3, stride = 1,padding = 8, dilation = 8), norm_layer(ngf * 4, affine = True)]
        #Dilated conv10
        model += [nn.Conv2d(ngf * 4, ngf * 4,kernel_size = 3, stride = 1,padding = 16, dilation = 16), norm_layer(ngf * 4, affine = True)]
        #----------------------
        #conv11
        model += [nn.Conv2d(ngf * 4, ngf * 4,kernel_size = 3, stride = 1,padding = 1), norm_layer(ngf * 4, affine = True)]
        #conv12
        model += [nn.Conv2d(ngf * 4, ngf * 4,kernel_size = 3, stride = 1,padding = 1), norm_layer(ngf * 4, affine = True)]
        #deconv
        model += [nn.ConvTranspose2d(ngf * 4, ngf * 2,kernel_size = 4, stride = 2,padding = 1), norm_layer(ngf * 2, affine = True)]
        #conv
        model += [nn.Conv2d(ngf * 2, ngf * 2,kernel_size = 3, stride = 1,padding = 1), norm_layer(ngf * 2, affine = True)]
        #deconv  
        model += [nn.ConvTranspose2d(ngf * 2, ngf * 1,kernel_size = 4, stride = 2,padding = 1), norm_layer(ngf, affine = True)]
        #conv
        model += [nn.Conv2d(ngf * 1, int(ngf / 2),kernel_size = 3, stride = 1,padding = 1), norm_layer(int(ngf/ 2 ), affine = True)]
        #output
        model += [nn.Conv2d(int(ngf / 2) , output_nc,kernel_size = 3, stride = 1,padding = 1), norm_layer(output_nc, affine = True)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
      if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
        return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
      else:
        return self.model(input)

#元々のジェネレータの作り。参考にすべし
class Old_ResnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, gpu_ids=[]):
        assert(n_blocks >= 0)
        super(Old_ResnetGenerator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.gpu_ids = gpu_ids

        #入力層
        #model = [nn.Conv2d(input_nc, ngf, kernel_size=5, stride=1, padding=2,dilation=1),
        #         norm_layer(ngf, affine=True),
        #         nn.ReLU(True)]
        model = [nn.Conv2d(input_nc, ngf, kernel_size=5, stride=1, padding=2,dilation=1),
                 nn.ReLU(True),
                 nn.Conv2d(input_nc, ngf, kernel_size=5, stride=1, padding=2,dilation=1),
                 nn.ReLU(True)]
        

        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2**i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                                stride=2, padding=1),
                      norm_layer(ngf * mult * 2, affine=True),
                      nn.ReLU(True)]

        mult = 2**n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, 'zero', norm_layer=norm_layer, use_dropout=use_dropout)]

        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1),
                      norm_layer(int(ngf * mult / 2), affine=True),
                      nn.ReLU(True)]

        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=3)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
      if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
        return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
      else:
        return self.model(input)



#256 512 64で行きたい
class Global_Discriminator(nn.Module):
  def __init__(self, input_nc, output_nc, ndf= 64, gpu_ids=[] ):
      super(Global_Discriminator, self).__init__()
      self.gpu_ids = gpu_ids
      self.input_nc = input_nc
      self.output_nc = output_nc
      self.ndf =ndf

      #1
      model = [nn.Conv2d(input_nc, ndf, kernel_size=5, stride=2, padding=2,dilation=1),
                
                nn.ReLU(True)]
      #conv2
      model += [nn.Conv2d(ndf, ndf * 2, kernel_size=5, stride=2, padding=2,dilation=1)]
      #conv2
      model += [nn.Conv2d(ndf * 2, ndf * 4, kernel_size=5, stride=2, padding=2,dilation=1)]
      model += [nn.Conv2d(ndf * 4, ndf * 8, kernel_size=5, stride=2, padding=2,dilation=1)]

      model += [nn.Conv2d(ndf * 8, ndf * 8, kernel_size=5, stride=2, padding=2,dilation=1)]
      model += [nn.Conv2d(ndf * 8, ndf * 8, kernel_size=5, stride=2, padding=2,dilation=1)]

      #FullConvolution層  
      model += [nn.Conv2d(ndf * 8, output_nc, 4, 1)]

      #model += [nn.Linear(512 * 4 * 4, output_nc)]

      #1024次元にしたい
      self.model = nn.Sequential(*model)


  def forward(self, input):
    #上の役割はGPUによる並列処理
    if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
        return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
    else:
        return self.model(input)


class Local_Discriminator(nn.Module):
  def __init__(self, input_nc, output_nc ,ndf= 64,gpu_ids=[] ):
      super(Local_Discriminator, self).__init__()
      self.gpu_ids = gpu_ids
      self.input_nc = input_nc
      self.output_nc = output_nc
      self.ndf = ndf

      #1
      model = [nn.Conv2d(input_nc, ndf, kernel_size=5, stride=2, padding=2,dilation=1),
                
                nn.ReLU(True)]
      #conv2
      model += [nn.Conv2d(ndf, ndf * 2, kernel_size=5, stride=2, padding=2,dilation=1)]
      #conv2
      model += [nn.Conv2d(ndf * 2, ndf * 4, kernel_size=5, stride=2, padding=2,dilation=1)]
      model += [nn.Conv2d(ndf * 4, ndf * 8, kernel_size=5, stride=2, padding=2,dilation=1)]

      model += [nn.Conv2d(ndf * 8, ndf * 8, kernel_size=5, stride=2, padding=2,dilation=1)]

      #FullConvolution層
      model += [nn.Conv2d(ndf * 8, output_nc, 4, 1)]

      self.model = nn.Sequential(*model)

  def forward(self, input):
    if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
        return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
    else:
        return self.model(input)
'''
class Edge_Detector(nn.Module):
  def __init__(self, input_nc,output_nc,gpu_ids=[],tensor=torch.FloatTensor):
      super(Edge_Detector,self).__init__()
      self.gpu_ids = gpu_ids
      self.input_nc = input_nc
      self.output_nc = output_nc
      self.dtype = torch.float
      self.device = torch.device("cuda:0")
      self.tensor = tensor

      self.kernel = torch.tensor([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]],device="cuda:0" )
      self.gray_kernel = torch.tensor([0.299, 0.587, 0.114],device="cuda:0").reshape(1, 3, 1, 1)
      #self.edge_k = torch.as_tensor(kernel.reshape(1,1,3,3))#.to(self.device,self.dtype)
      #self.gray_kernel = np.array([0.299, 0.587, 0.114]).reshape(1, 3, 1, 1)
      model = [F.Conv2d()]

  def forward(self, input):
    #モデルの形成はforwardでやってみる
    #kernel = np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]])

    edge_k = self.kernel.reshape(1,1,3,3)#.to(self.device,self.dtype)
    color = input#.to(self.dtype)
    #gray_kernel = np.array([0.299, 0.587, 0.114]).reshape(1, 3, 1, 1)
    #gray_k = torch.as_tensor(self.gray_kernel)#.to(self.device,self.dtype)

    gray = torch.sum(color * self.gray_kernel,dim = 1,keepdim = True)
    #model = nn.Sequential()

    model = [F.Conv2d(gray,edge_k  ,padding=1)]
    self.model = nn.Sequential(*model)

    if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
        return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
    else:
      return self.model(input)
'''

#形質としてはLocalDiscriminatorと同じ形を目指す
class Edge_Discriminator(nn.Module):
  def __init__(self, input_nc, output_nc ,ndf= 64,gpu_ids=[] ):
      super(Edge_Discriminator, self).__init__()
      self.gpu_ids = gpu_ids
      self.input_nc = input_nc
      self.output_nc = output_nc
      self.ndf = ndf

      #1
      model = [nn.Conv2d(input_nc, ndf, kernel_size=5, stride=2, padding=2,dilation=1),
                
                nn.ReLU(True)]
      #conv2
      model += [nn.Conv2d(ndf, ndf * 2, kernel_size=5, stride=2, padding=2,dilation=1)]
      #conv2
      model += [nn.Conv2d(ndf * 2, ndf * 4, kernel_size=5, stride=2, padding=2,dilation=1)]
      model += [nn.Conv2d(ndf * 4, ndf * 8, kernel_size=5, stride=2, padding=2,dilation=1)]

      model += [nn.Conv2d(ndf * 8, ndf * 8, kernel_size=5, stride=2, padding=2,dilation=1)]
      model += [nn.Conv2d(ndf * 8, ndf * 8, kernel_size=5, stride=2, padding=2,dilation=1)]

      #1024次元にしたい
      #FullConvolution層  
      model += [nn.Conv2d(ndf * 8, output_nc, 2, 1)] #カーネルを4→2にすれば良いらしいが不明

      self.model = nn.Sequential(*model)

  def forward(self, input):
    if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
        return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
    else:
        return self.model(input)

def load_tensor(__input):
    
    array = np.asarray(__input, np.float32).transpose([2, 0, 1]) / 255.0
    tensor = torch.as_tensor(np.expand_dims(array, axis=0))  # rank 4
    return tensor

def edge_detection(__input):
    kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], np.float32)  # convolution filter
    with torch.no_grad():
        # [out_ch, in_ch, .., ..] : channel wiseに計算
        edge_k = torch.as_tensor(kernel.reshape(1, 1, 3, 3))

        # エッジ検出はグレースケール化してからやる
        color = __input  # color image [1, 3, H, W]
        gray_kernel = np.array([0.299, 0.587, 0.114], np.float32).reshape(1, 3, 1, 1)  # color -> gray kernel
        gray_k = torch.as_tensor(gray_kernel)
        
        gray = torch.sum(color * gray_k, dim=1, keepdim=True)  # grayscale image [1, 1, H, W]

        # エッジ検出
        edge_image = F.conv2d(gray, edge_k, padding=1)
        return edge_image

# Define a resnet block
class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout)



    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout):
        conv_block = []
        p = 0
        assert(padding_type == 'zero')
        p = 1

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim, affine=True),
                       nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim, affine=True)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


# Defines the PatchGAN discriminator.
class NLayerDiscriminator(nn.Module):
  def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False, gpu_ids=[]):
    super(NLayerDiscriminator, self).__init__()
    self.gpu_ids = gpu_ids

    kw = 4
    padw = int(np.ceil((kw-1)/2))
    sequence = [
        nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
        nn.LeakyReLU(0.2, True)
    ]

    nf_mult = 1
    nf_mult_prev = 1
    for n in range(1, n_layers):
        nf_mult_prev = nf_mult
        nf_mult = min(2**n, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2,
                      padding=padw), norm_layer(ndf * nf_mult,
                                                affine=True), nn.LeakyReLU(0.2, True)
        ]

    nf_mult_prev = nf_mult
    nf_mult = min(2**n_layers, 8)
    sequence += [
        nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1,
                  padding=padw), norm_layer(ndf * nf_mult,
                                            affine=True), nn.LeakyReLU(0.2, True)
    ]

    sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]

    if use_sigmoid:
        sequence += [nn.Sigmoid()]

    self.model = nn.Sequential(*sequence)

  def forward(self, input):
    if len(self.gpu_ids)  and isinstance(input.data, torch.cuda.FloatTensor):
      return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
    else:
      return self.model(input)

  
        #n_downsampling = 2
        #for i in range(n_downsampling):
        #    mult = 2**i # 2,4
        #    model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
        #                        stride=2, padding=1),
        #              norm_layer(ngf * mult * 2, affine=True),
        #              nn.ReLU(True)]



        #mult = 2**n_downsampling
        #for i in range(n_blocks):
        #    model += [ResnetBlock(ngf * mult, 'zero', norm_layer=norm_layer, use_dropout=use_dropout)]##
#
#        for i in range(n_downsampling):
#            mult = 2**(n_downsampling - i)
#            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
#                                         kernel_size=3, stride=2,
#                                         padding=1, output_padding=1),
#                      norm_layer(int(ngf * mult / 2), affine=True),
#                      nn.ReLU(True)]
#
#        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=3)]
#        model += [nn.Tanh()]

#        self.model = nn.Sequential(*model)
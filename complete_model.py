#!/usr/bin/env python3
"""
SinGAN: Learning a Generative Model from a Single Natural Image
Consolidated implementation for Google Colab

Based on the original SinGAN implementation:
https://github.com/tamarott/SinGAN
"""

import os
import sys
import random
import math
import numpy as np
import argparse
from types import SimpleNamespace
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage import io as img
from skimage import color, morphology, filters
from sklearn.cluster import KMeans
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.utils as vutils
import torchvision.transforms as transforms
from torch.nn.functional import adaptive_avg_pool2d
import scipy
from scipy import linalg
from scipy.ndimage import filters as scipy_filters
from scipy.ndimage import measurements, interpolation
import imageio

# For Google Colab compatibility
try:
    from google.colab import files
    IN_COLAB = True
except:
    IN_COLAB = False

try:
    from tqdm import tqdm
except ImportError:
    # If tqdm is not available, provide a mock version of it
    def tqdm(x): return x

# Initialize device
device = torch.device("cuda:0" if (torch.cuda.is_available() and not torch.cuda.is_initialized()) else "cpu")
print(f"Using device: {device}")

# Create directories for input/output
os.makedirs('Input/Images', exist_ok=True)
os.makedirs('Input/Harmonization', exist_ok=True)
os.makedirs('Input/Editing', exist_ok=True)
os.makedirs('Input/Paint', exist_ok=True)
os.makedirs('Output', exist_ok=True)

#############################
# Model classes (from models.py)
#############################

class ConvBlock(nn.Sequential):
    def __init__(self, in_channel, out_channel, ker_size, padd, stride):
        super(ConvBlock, self).__init__()
        self.add_module('conv', nn.Conv2d(in_channel, out_channel, kernel_size=ker_size, stride=stride, padding=padd))
        self.add_module('norm', nn.BatchNorm2d(out_channel))
        self.add_module('LeakyRelu', nn.LeakyReLU(0.2, inplace=False))

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        # Use non-in-place operations for weight initialization
        m.weight.data = torch.normal(0.0, 0.02, size=m.weight.data.shape, device=m.weight.device)
    elif classname.find('Norm') != -1:
        # Use non-in-place operations for weight and bias initialization
        m.weight.data = torch.normal(1.0, 0.02, size=m.weight.data.shape, device=m.weight.device)
        m.bias.data = torch.zeros_like(m.bias.data)
   
class WDiscriminator(nn.Module):
    def __init__(self, opt):
        super(WDiscriminator, self).__init__()
        self.is_cuda = torch.cuda.is_available()
        N = int(opt.nfc)
        self.head = ConvBlock(opt.nc_im, N, opt.ker_size, opt.padd_size, 1)
        self.body = nn.Sequential()
        for i in range(opt.num_layer-2):
            N = int(opt.nfc/pow(2, (i+1)))
            block = ConvBlock(max(2*N, opt.min_nfc), max(N, opt.min_nfc), opt.ker_size, opt.padd_size, 1)
            self.body.add_module('block%d' % (i+1), block)
        self.tail = nn.Conv2d(max(N, opt.min_nfc), 1, kernel_size=opt.ker_size, stride=1, padding=opt.padd_size)

    def forward(self, x):
        x = self.head(x)
        x = self.body(x)
        x = self.tail(x)
        return x

class GeneratorConcatSkip2CleanAdd(nn.Module):
    def __init__(self, opt):
        super(GeneratorConcatSkip2CleanAdd, self).__init__()
        self.is_cuda = torch.cuda.is_available()
        N = opt.nfc
        self.head = ConvBlock(opt.nc_im, N, opt.ker_size, opt.padd_size, 1)
        self.body = nn.Sequential()
        for i in range(opt.num_layer-2):
            N = int(opt.nfc/pow(2, (i+1)))
            block = ConvBlock(max(2*N, opt.min_nfc), max(N, opt.min_nfc), opt.ker_size, opt.padd_size, 1)
            self.body.add_module('block%d' % (i+1), block)
        self.tail = nn.Sequential(
            nn.Conv2d(max(N, opt.min_nfc), opt.nc_im, kernel_size=opt.ker_size, stride=1, padding=opt.padd_size),
            nn.Tanh()
        )
    
    def forward(self, x, y):
        x = self.head(x)
        x = self.body(x)
        x = self.tail(x)
        ind = int((y.shape[2]-x.shape[2])/2)
        y = y[:, :, ind:(y.shape[2]-ind), ind:(y.shape[3]-ind)]
        # Create a new tensor instead of in-place addition
        return x + y.clone()

#############################
# InceptionV3 for SIFID (from SIFID/inception.py)
#############################

class InceptionV3(nn.Module):
    """Pretrained InceptionV3 network returning feature maps"""

    # Index of default block of inception to return,
    # corresponds to output of final average pooling
    DEFAULT_BLOCK_INDEX = 3

    # Maps feature dimensionality to their output blocks indices
    BLOCK_INDEX_BY_DIM = {
        64: 0,   # First max pooling features
        192: 1,  # Second max pooling featurs
        768: 2,  # Pre-aux classifier features
        2048: 3  # Final average pooling features
    }

    def __init__(self,
                 output_blocks=[DEFAULT_BLOCK_INDEX],
                 resize_input=False,
                 normalize_input=True,
                 requires_grad=False):
        """Build pretrained InceptionV3

        Parameters
        ----------
        output_blocks : list of int
            Indices of blocks to return features of. Possible values are:
                - 0: corresponds to output of first max pooling
                - 1: corresponds to output of second max pooling
                - 2: corresponds to output which is fed to aux classifier
                - 3: corresponds to output of final average pooling
        resize_input : bool
            If true, bilinearly resizes input to width and height 299 before
            feeding input to model. As the network without fully connected
            layers is fully convolutional, it should be able to handle inputs
            of arbitrary size, so resizing might not be strictly needed
        normalize_input : bool
            If true, scales the input from range (0, 1) to the range the
            pretrained Inception network expects, namely (-1, 1)
        requires_grad : bool
            If true, parameters of the model require gradient. Possibly useful
            for finetuning the network
        """
        super(InceptionV3, self).__init__()

        self.resize_input = resize_input
        self.normalize_input = normalize_input
        self.output_blocks = sorted(output_blocks)
        self.last_needed_block = max(output_blocks)

        assert self.last_needed_block <= 3, \
            'Last possible output block index is 3'

        self.blocks = nn.ModuleList()

        inception = torchvision.models.inception_v3(pretrained=True)

        # Block 0: input to maxpool1
        block0 = [
            inception.Conv2d_1a_3x3,
            inception.Conv2d_2a_3x3,
            inception.Conv2d_2b_3x3,
            ]
        self.blocks.append(nn.Sequential(*block0))

        # Block 1: maxpool1 to maxpool2
        if self.last_needed_block >= 1:
            block1 = [
                nn.MaxPool2d(kernel_size=3, stride=2),
                inception.Conv2d_3b_1x1,
                inception.Conv2d_4a_3x3,
            ]
            self.blocks.append(nn.Sequential(*block1))

        # Block 2: maxpool2 to aux classifier
        if self.last_needed_block >= 2:
            block2 = [
                nn.MaxPool2d(kernel_size=3, stride=2),
                inception.Mixed_5b,
                inception.Mixed_5c,
                inception.Mixed_5d,
                inception.Mixed_6a,
                inception.Mixed_6b,
                inception.Mixed_6c,
                inception.Mixed_6d,
                inception.Mixed_6e,
            ]
            self.blocks.append(nn.Sequential(*block2))

        # Block 3: aux classifier to final avgpool
        if self.last_needed_block >= 3:
            block3 = [
                inception.Mixed_7a,
                inception.Mixed_7b,
                inception.Mixed_7c,
            ]
            self.blocks.append(nn.Sequential(*block3))

        if self.last_needed_block >= 4:
            block4 = [
                nn.AdaptiveAvgPool2d(output_size=(1, 1))
            ]
            self.blocks.append(nn.Sequential(*block4))

        for param in self.parameters():
            param.requires_grad = requires_grad

    def forward(self, inp):
        """Get Inception feature maps

        Parameters
        ----------
        inp : torch.autograd.Variable
            Input tensor of shape Bx3xHxW. Values are expected to be in
            range (0, 1)

        Returns
        -------
        List of torch.autograd.Variable, corresponding to the selected output
        block, sorted ascending by index
        """
        outp = []
        x = inp

        if self.resize_input:
            x = F.interpolate(x, size=(299, 299), mode='bilinear', align_corners=False)

        if self.normalize_input:
            x = 2 * x - 1  # Scale from range (0, 1) to range (-1, 1)

        for idx, block in enumerate(self.blocks):
            x = block(x)
            if idx in self.output_blocks:
                outp.append(x)

            if idx == self.last_needed_block:
                break

        return outp

#############################
# Utility functions (from functions.py)
#############################

def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)

def norm(x):
    out = (x - 0.5) * 2
    return out.clamp(-1, 1)

def convert_image_np(inp):
    # Create a defensive copy to avoid in-place modifications
    inp = inp.clone().detach()
    
    if inp.shape[1] == 3:
        inp = denorm(inp)
        inp = inp[-1, :, :, :]
        inp = inp.to('cpu').numpy().transpose((1, 2, 0))
    else:
        inp = denorm(inp)
        inp = inp[-1, -1, :, :]
        inp = inp.to('cpu').numpy().transpose((0, 1))
    inp = np.clip(inp, 0, 1)
    return inp

def generate_noise(size, num_samp=1, device='cuda', type='gaussian', scale=1):
    if type == 'gaussian':
        noise = torch.randn(num_samp, size[0], round(size[1]/scale), round(size[2]/scale), device=device)
        noise = upsampling(noise, size[1], size[2])
    if type == 'gaussian_mixture':
        noise1 = torch.randn(num_samp, size[0], size[1], size[2], device=device) + 5
        noise2 = torch.randn(num_samp, size[0], size[1], size[2], device=device)
        noise = noise1 + noise2
    if type == 'uniform':
        noise = torch.randn(num_samp, size[0], size[1], size[2], device=device)
    return noise

def plot_learning_curves(G_loss, D_loss, epochs, label1, label2, name):
    fig, ax = plt.subplots(1)
    n = np.arange(0, epochs)
    plt.plot(n, G_loss, n, D_loss)
    plt.xlabel('epochs')
    plt.legend([label1, label2], loc='upper right')
    plt.savefig('%s.png' % name)
    plt.close(fig)

def plot_learning_curve(loss, epochs, name):
    fig, ax = plt.subplots(1)
    n = np.arange(0, epochs)
    plt.plot(n, loss)
    plt.ylabel('loss')
    plt.xlabel('epochs')
    plt.savefig('%s.png' % name)
    plt.close(fig)

def upsampling(im, sx, sy):
    m = nn.Upsample(size=[round(sx), round(sy)], mode='bilinear', align_corners=True)
    return m(im)

def reset_grads(model, require_grad):
    for p in model.parameters():
        p.requires_grad_(require_grad)
    return model

def calc_gradient_penalty(netD, real_data, fake_data, LAMBDA, device):
    # Use detached copies to prevent in-place operations from affecting the computational graph
    real_data = real_data.detach().clone()
    fake_data = fake_data.detach().clone()
    
    alpha = torch.rand(1, 1)
    alpha = alpha.expand(real_data.size())
    alpha = alpha.to(device)

    # Create interpolated images without in-place operations
    interpolates = alpha * real_data + ((1 - alpha) * fake_data)
    interpolates = interpolates.to(device)
    interpolates = torch.autograd.Variable(interpolates, requires_grad=True)

    disc_interpolates = netD(interpolates)

    gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
    
    # Calculate penalty without in-place operations
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    return gradient_penalty

def read_image(opt):
    x = img.imread('%s/%s' % (opt.input_dir, opt.input_name))
    x = np2torch(x, opt)
    x = x[:, 0:3, :, :]
    return x

def read_image_dir(dir, opt):
    x = img.imread('%s' % (dir))
    x = np2torch(x, opt)
    x = x[:, 0:3, :, :]
    return x

def np2torch(x, opt):
    if opt.nc_im == 3:
        x = x[:, :, :, None]
        x = x.transpose((3, 2, 0, 1))/255
    else:
        x = color.rgb2gray(x)
        x = x[:, :, None, None]
        x = x.transpose(3, 2, 0, 1)
    x = torch.from_numpy(x)
    
    # Check if CUDA is actually available before trying to use it
    cuda_available = torch.cuda.is_available() and not opt.not_cuda
    
    if cuda_available:
        try:
            x = x.to(device)
            x = x.type(torch.cuda.FloatTensor)
        except RuntimeError:
            print("Warning: Could not use CUDA. Falling back to CPU.")
            opt.not_cuda = True
            cuda_available = False
    
    if not cuda_available:
        x = x.type(torch.FloatTensor)
    
    x = norm(x)
    return x

def torch2uint8(x):
    x = x[0, :, :, :]
    x = x.permute((1, 2, 0))
    x = 255 * denorm(x)
    x = x.cpu().numpy()
    x = x.astype(np.uint8)
    return x

def save_networks(netG, netD, z, opt):
    torch.save(netG.state_dict(), '%s/netG.pth' % (opt.outf))
    torch.save(netD.state_dict(), '%s/netD.pth' % (opt.outf))
    torch.save(z, '%s/z_opt.pth' % (opt.outf))

def adjust_scales2image(real_, opt):
    opt.num_scales = math.ceil((math.log(math.pow(opt.min_size / (min(real_.shape[2], real_.shape[3])), 1), opt.scale_factor_init))) + 1
    scale2stop = math.ceil(math.log(min([opt.max_size, max([real_.shape[2], real_.shape[3]])]) / max([real_.shape[2], real_.shape[3]]), opt.scale_factor_init))
    opt.stop_scale = opt.num_scales - scale2stop
    opt.scale1 = min(opt.max_size / max([real_.shape[2], real_.shape[3]]), 1)
    real = imresize(real_, scale_factor=opt.scale1)
    opt.scale_factor = math.pow(opt.min_size/(min(real.shape[2], real.shape[3])), 1/(opt.stop_scale))
    scale2stop = math.ceil(math.log(min([opt.max_size, max([real_.shape[2], real_.shape[3]])]) / max([real_.shape[2], real_.shape[3]]), opt.scale_factor_init))
    opt.stop_scale = opt.num_scales - scale2stop
    return real

def adjust_scales2image_SR(real_, opt):
    opt.min_size = 18
    opt.num_scales = int((math.log(opt.min_size / min(real_.shape[2], real_.shape[3]), opt.scale_factor_init))) + 1
    scale2stop = int(math.log(min(opt.max_size, max(real_.shape[2], real_.shape[3])) / max(real_.shape[0], real_.shape[3]), opt.scale_factor_init))
    opt.stop_scale = opt.num_scales - scale2stop
    opt.scale1 = min(opt.max_size / max([real_.shape[2], real_.shape[3]]), 1)
    real = imresize(real_, scale_factor=opt.scale1)
    opt.scale_factor = math.pow(opt.min_size/(min(real.shape[2], real.shape[3])), 1/(opt.stop_scale))
    scale2stop = int(math.log(min(opt.max_size, max(real_.shape[2], real_.shape[3])) / max(real_.shape[0], real_.shape[3]), opt.scale_factor_init))
    opt.stop_scale = opt.num_scales - scale2stop
    return real

def creat_reals_pyramid(real, reals, opt):
    real = real[:, 0:3, :, :]
    for i in range(0, opt.stop_scale+1, 1):
        scale = math.pow(opt.scale_factor, opt.stop_scale-i)
        curr_real = imresize(real, scale_factor=scale)
        reals.append(curr_real)
    return reals

def load_trained_pyramid(opt, mode_='train'):
    mode = opt.mode
    opt.mode = 'train'
    if (mode == 'animation_train') | (mode == 'SR_train') | (mode == 'paint_train'):
        opt.mode = mode
    dir = generate_dir2save(opt)
    if os.path.exists(dir):
        Gs = torch.load('%s/Gs.pth' % dir, map_location=device)
        Zs = torch.load('%s/Zs.pth' % dir, map_location=device)
        reals = torch.load('%s/reals.pth' % dir, map_location=device)
        NoiseAmp = torch.load('%s/NoiseAmp.pth' % dir, map_location=device)
    else:
        print('No appropriate trained model is found, please train first.')
        exit()
    opt.mode = mode
    return Gs, Zs, reals, NoiseAmp

def generate_in2coarsest(reals, scale_v, scale_h, opt):
    real = reals[opt.gen_start_scale]
    real_down = upsampling(real, scale_v * real.shape[2], scale_h * real.shape[3])
    if opt.gen_start_scale == 0:
        in_s = torch.full(real_down.shape, 0, device=device)
    else:
        in_s = upsampling(real_down, real_down.shape[2], real_down.shape[3])
    return in_s

def generate_dir2save(opt):
    dir2save = None
    if (opt.mode == 'train') | (opt.mode == 'SR_train'):
        dir2save = 'TrainedModels/%s/scale_factor=%f,alpha=%d' % (opt.input_name[:-4], opt.scale_factor_init, opt.alpha)
    elif (opt.mode == 'animation_train'):
        dir2save = 'TrainedModels/%s/scale_factor=%f_noise_padding' % (opt.input_name[:-4], opt.scale_factor_init)
    elif (opt.mode == 'paint_train'):
        dir2save = 'TrainedModels/%s/scale_factor=%f_paint/start_scale=%d' % (opt.input_name[:-4], opt.scale_factor_init, opt.paint_start_scale)
    elif opt.mode == 'random_samples':
        dir2save = '%s/RandomSamples/%s/gen_start_scale=%d' % (opt.out, opt.input_name[:-4], opt.gen_start_scale)
    elif opt.mode == 'random_samples_arbitrary_sizes':
        dir2save = '%s/RandomSamples_ArbitrerySizes/%s/scale_v=%f_scale_h=%f' % (opt.out, opt.input_name[:-4], opt.scale_v, opt.scale_h)
    elif opt.mode == 'animation':
        dir2save = '%s/Animation/%s' % (opt.out, opt.input_name[:-4])
    elif opt.mode == 'SR':
        dir2save = '%s/SR/%s' % (opt.out, opt.sr_factor)
    elif opt.mode == 'harmonization':
        dir2save = '%s/Harmonization/%s/%s_out' % (opt.out, opt.input_name[:-4], opt.ref_name[:-4])
    elif opt.mode == 'editing':
        dir2save = '%s/Editing/%s/%s_out' % (opt.out, opt.input_name[:-4], opt.ref_name[:-4])
    elif opt.mode == 'paint2image':
        dir2save = '%s/Paint2image/%s/%s_out' % (opt.out, opt.input_name[:-4], opt.ref_name[:-4])
        if opt.quantization_flag:
            dir2save = '%s_quantized' % dir2save
    return dir2save

def post_config(opt):
    # init fixed parameters
    cuda_available = torch.cuda.is_available() and not opt.not_cuda
    if cuda_available:
        try:
            # Try a small CUDA operation to verify it works
            test_tensor = torch.zeros(1).cuda()
            del test_tensor
            opt.device = torch.device("cuda:0")
        except RuntimeError:
            print("Warning: CUDA initialization failed. Using CPU instead.")
            opt.device = torch.device("cpu")
            opt.not_cuda = True
    else:
        opt.device = torch.device("cpu")
        if not opt.not_cuda and not torch.cuda.is_available():
            print("Warning: CUDA is not available. Using CPU instead.")
            opt.not_cuda = True
    
    opt.niter_init = opt.niter
    opt.noise_amp_init = opt.noise_amp
    opt.nfc_init = opt.nfc
    opt.min_nfc_init = opt.min_nfc
    opt.scale_factor_init = opt.scale_factor
    opt.out_ = 'TrainedModels/%s/scale_factor=%f/' % (opt.input_name[:-4], opt.scale_factor)
    if opt.mode == 'SR':
        opt.alpha = 100

    if opt.manualSeed is None:
        opt.manualSeed = random.randint(1, 10000)
    print("Random Seed: ", opt.manualSeed)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    
    # Only show this warning if the user specifically asked for CUDA
    if not opt.not_cuda and not torch.cuda.is_available():
        print("WARNING: You specified to use CUDA, but it's not available. Running on CPU instead.")
    
    return opt

def calc_init_scale(opt):
    in_scale = math.pow(1/2, 1/3)
    iter_num = round(math.log(1 / opt.sr_factor, in_scale))
    in_scale = pow(opt.sr_factor, 1 / iter_num)
    return in_scale, iter_num

def quant(prev, device):
    arr = prev.reshape((-1, 3)).cpu()
    kmeans = KMeans(n_clusters=5, random_state=0).fit(arr)
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_
    x = centers[labels]
    x = torch.from_numpy(x)
    x = x.to(device)
    x = x.type(torch.cuda.FloatTensor) if torch.cuda.is_available() else x.type(torch.FloatTensor)
    x = x.view(prev.shape)
    return x, centers

def quant2centers(paint, centers):
    arr = paint.reshape((-1, 3)).cpu()
    kmeans = KMeans(n_clusters=5, init=centers, n_init=1).fit(arr)
    labels = kmeans.labels_
    x = centers[labels]
    x = torch.from_numpy(x)
    x = x.to(device)
    x = x.type(torch.cuda.FloatTensor) if torch.cuda.is_available() else x.type(torch.FloatTensor)
    x = x.view(paint.shape)
    return x

def dilate_mask(mask, opt):
    if opt.mode == "harmonization":
        element = morphology.disk(radius=7)
    if opt.mode == "editing":
        element = morphology.disk(radius=20)
    mask = torch2uint8(mask)
    mask = mask[:, :, 0]
    mask = morphology.binary_dilation(mask, selem=element)
    mask = filters.gaussian(mask, sigma=5)
    nc_im = opt.nc_im
    opt.nc_im = 1
    mask = np2torch(mask, opt)
    opt.nc_im = nc_im
    mask = mask.expand(1, 3, mask.shape[2], mask.shape[3])
    plt.imsave('%s/%s_mask_dilated.png' % (opt.ref_dir, opt.ref_name[:-4]), convert_image_np(mask), vmin=0, vmax=1)
    mask = (mask-mask.min())/(mask.max()-mask.min())
    return mask

#############################
# Image resizing functions (from imresize.py)
#############################

def imresize(im, scale_factor=None, output_shape=None, kernel=None, antialiasing=True, kernel_shift_flag=False):
    # First convert to numpy
    im = torch2uint8(im)
    
    # Check parameters
    if isinstance(output_shape, argparse.Namespace):
        # This is likely a mistake - opt is being passed as output_shape
        # Extract relevant fields or just use scale_factor instead
        print("Warning: Namespace object detected as output_shape; converting to None")
        output_shape = None
    
    # Apply resizing function
    im_resized = imresize_in(im, scale_factor=scale_factor, output_shape=output_shape, 
                             kernel=kernel, antialiasing=antialiasing, kernel_shift_flag=kernel_shift_flag)
    
    # Convert back to torch tensor
    im_resized = np2torch(im_resized, SimpleNamespace(nc_im=3, not_cuda=not torch.cuda.is_available()))
    return im_resized

def imresize_to_shape(im, output_shape, opt):
    im = torch2uint8(im)
    im = imresize_in(im, output_shape=output_shape)
    im = np2torch(im, opt)
    return im

def imresize_in(im, scale_factor=None, output_shape=None, kernel=None, antialiasing=True, kernel_shift_flag=False):
    # First standardize values and fill missing arguments (if needed) by deriving scale from output shape or vice versa
    if isinstance(output_shape, argparse.Namespace):
        raise TypeError("output_shape should be a shape tuple/list, not a Namespace object")
    scale_factor, output_shape = fix_scale_and_size(im.shape, output_shape, scale_factor)

    # For a given numeric kernel case, just do convolution and sub-sampling (downscaling only)
    if type(kernel) == np.ndarray and scale_factor[0] <= 1:
        return numeric_kernel(im, kernel, scale_factor, output_shape, kernel_shift_flag)

    # Choose interpolation method, each method has the matching kernel size
    method, kernel_width = {
        "cubic": (cubic, 4.0),
        "lanczos2": (lanczos2, 4.0),
        "lanczos3": (lanczos3, 6.0),
        "box": (box, 1.0),
        "linear": (linear, 2.0),
        None: (cubic, 4.0)  # set default interpolation method as cubic
    }.get(kernel)

    # Antialiasing is only used when downscaling
    antialiasing *= (scale_factor[0] < 1)

    # Sort indices of dimensions according to scale of each dimension. since we are going dim by dim this is efficient
    sorted_dims = np.argsort(np.array(scale_factor)).tolist()

    # Iterate over dimensions to calculate local weights for resizing and resize each time in one direction
    out_im = np.copy(im)
    for dim in sorted_dims:
        # No point doing calculations for scale-factor 1. nothing will happen anyway
        if scale_factor[dim] == 1.0:
            continue

        # for each coordinate (along 1 dim), calculate which coordinates in the input image affect its result and the
        # weights that multiply the values there to get its result.
        weights, field_of_view = contributions(im.shape[dim], output_shape[dim], scale_factor[dim],
                                           method, kernel_width, antialiasing)

        # Use the affecting position values and the set of weights to calculate the result of resizing along this 1 dim
        out_im = resize_along_dim(out_im, dim, weights, field_of_view)

    return out_im

def fix_scale_and_size(input_shape, output_shape, scale_factor):
    # First fixing the scale-factor (if given) to be standardized the function expects (a list of scale factors in the
    # same size as the number of input dimensions)
    if scale_factor is not None:
        # By default, if scale-factor is a scalar we assume 2d resizing and duplicate it.
        if np.isscalar(scale_factor):
            scale_factor = [scale_factor, scale_factor]

        # We extend the size of scale-factor list to the size of the input by assigning 1 to all the unspecified scales
        scale_factor = list(scale_factor)
        scale_factor.extend([1] * (len(input_shape) - len(scale_factor)))

    # Fixing output-shape (if given): extending it to the size of the input-shape, by assigning the original input-size
    # to all the unspecified dimensions
    if output_shape is not None:
        output_shape = list(np.uint(np.array(output_shape))) + list(input_shape[len(output_shape):])

    # Dealing with the case of non-give scale-factor, calculating according to output-shape. note that this is
    # sub-optimal, because there can be different scales to the same output-shape.
    if scale_factor is None:
        scale_factor = 1.0 * np.array(output_shape) / np.array(input_shape)

    # Dealing with missing output-shape. calculating according to scale-factor
    if output_shape is None:
        output_shape = np.uint(np.ceil(np.array(input_shape) * np.array(scale_factor)))

    return scale_factor, output_shape

def contributions(in_length, out_length, scale, kernel, kernel_width, antialiasing):
    # This function calculates a set of 'filters' and a set of field_of_view that will later on be applied
    # such that each position from the field_of_view will be multiplied with a matching filter from the
    # 'weights' based on the interpolation method and the distance of the sub-pixel location from the pixel centers
    # around it. This is only done for one dimension of the image.

    # When anti-aliasing is activated (default and only for downscaling) the receptive field is stretched to size of
    # 1/sf. this means filtering is more 'low-pass filter'.
    fixed_kernel = (lambda arg: scale * kernel(scale * arg)) if antialiasing else kernel
    kernel_width *= 1.0 / scale if antialiasing else 1.0

    # These are the coordinates of the output image
    out_coordinates = np.arange(1, out_length+1)

    # These are the matching positions of the output-coordinates on the input image coordinates.
    # Best explained by example: say we have 4 horizontal pixels for HR and we downscale by SF=2 and get 2 pixels:
    # [1,2,3,4] -> [1,2]. Remember each pixel number is the middle of the pixel.
    # The scaling is done between the distances and not pixel numbers (the right boundary of pixel 4 is transformed to
    # the right boundary of pixel 2. pixel 1 in the small image matches the boundary between pixels 1 and 2 in the big
    # one and not to pixel 2. This means the position is not just multiplication of the old pos by scale-factor).
    # So if we measure distance from the left border, middle of pixel 1 is at distance d=0.5, border between 1 and 2 is
    # at d=1, and so on (d = p - 0.5).  we calculate (d_new = d_old / sf) which means:
    # (p_new-0.5 = (p_old-0.5) / sf)     ->          p_new = p_old/sf + 0.5 * (1-1/sf)
    match_coordinates = 1.0 * out_coordinates / scale + 0.5 * (1 - 1.0 / scale)

    # This is the left boundary to start multiplying the filter from, it depends on the size of the filter
    left_boundary = np.floor(match_coordinates - kernel_width / 2)

    # Kernel width needs to be enlarged because when covering has sub-pixel borders, it must 'see' the pixel centers
    # of the pixels it only covered a part from. So we add one pixel at each side to consider (weights can zeroize them)
    expanded_kernel_width = np.ceil(kernel_width) + 2

    # Determine a set of field_of_view for each each output position, these are the pixels in the input image
    # that the pixel in the output image 'sees'. We get a matrix whos horizontal dim is the output pixels (big) and the
    # vertical dim is the pixels it 'sees' (kernel_size + 2)
    field_of_view = np.squeeze(np.uint(np.expand_dims(left_boundary, axis=1) + np.arange(expanded_kernel_width) - 1))

    # Assign weight to each pixel in the field of view. A matrix whos horizontal dim is the output pixels and the
    # vertical dim is a list of weights matching to the pixel in the field of view (that are specified in
    # 'field_of_view')
    weights = fixed_kernel(1.0 * np.expand_dims(match_coordinates, axis=1) - field_of_view - 1)

    # Normalize weights to sum up to 1. be careful from dividing by 0
    sum_weights = np.sum(weights, axis=1)
    sum_weights[sum_weights == 0] = 1.0
    weights = 1.0 * weights / np.expand_dims(sum_weights, axis=1)

    # We use this mirror structure as a trick for reflection padding at the boundaries
    mirror = np.uint(np.concatenate((np.arange(in_length), np.arange(in_length - 1, -1, step=-1))))
    field_of_view = mirror[np.mod(field_of_view, mirror.shape[0])]

    # Get rid of  weights and pixel positions that are of zero weight
    non_zero_out_pixels = np.nonzero(np.any(weights, axis=0))
    weights = np.squeeze(weights[:, non_zero_out_pixels])
    field_of_view = np.squeeze(field_of_view[:, non_zero_out_pixels])

    # Final products are the relative positions and the matching weights, both are output_size X fixed_kernel_size
    return weights, field_of_view

def resize_along_dim(im, dim, weights, field_of_view):
    # To be able to act on each dim, we swap so that dim 0 is the wanted dim to resize
    tmp_im = np.swapaxes(im, dim, 0)

    # We add singleton dimensions to the weight matrix so we can multiply it with the big tensor we get for
    # tmp_im[field_of_view.T], (bsxfun style)
    weights = np.reshape(weights.T, list(weights.T.shape) + (np.ndim(im) - 1) * [1])

    # This is a bit of a complicated multiplication: tmp_im[field_of_view.T] is a tensor of order image_dims+1.
    # for each pixel in the output-image it matches the positions the influence it from the input image (along 1 dim
    # only, this is why it only adds 1 dim to the shape). We then multiply, for each pixel, its set of positions with
    # the matching set of weights. we do this by this big tensor element-wise multiplication (MATLAB bsxfun style:
    # matching dims are multiplied element-wise while singletons mean that the matching dim is all multiplied by the
    # same number
    tmp_out_im = np.sum(tmp_im[field_of_view.T] * weights, axis=0)

    # Finally we swap back the axes to the original order
    return np.swapaxes(tmp_out_im, dim, 0)

def numeric_kernel(im, kernel, scale_factor, output_shape, kernel_shift_flag):
    # See kernel_shift function to understand what this is
    if kernel_shift_flag:
        kernel = kernel_shift(kernel, scale_factor)

    # First run a correlation (convolution with flipped kernel)
    out_im = np.zeros_like(im)
    for channel in range(np.ndim(im)):
        out_im[:, :, channel] = scipy_filters.correlate(im[:, :, channel], kernel)

    # Then subsample and return
    return out_im[np.round(np.linspace(0, im.shape[0] - 1 / scale_factor[0], output_shape[0])).astype(int)[:, None],
                  np.round(np.linspace(0, im.shape[1] - 1 / scale_factor[1], output_shape[1])).astype(int), :]

def kernel_shift(kernel, sf):
    # There are two reasons for shifting the kernel:
    # 1. Center of mass is not in the center of the kernel which creates ambiguity. There is no possible way to know
    #    the degradation process included shifting so we always assume center of mass is center of the kernel.
    # 2. We further shift kernel center so that top left result pixel corresponds to the middle of the sfXsf first
    #    pixels. Default is for odd size to be in the middle of the first pixel and for even sized kernel to be at the
    #    top left corner of the first pixel. that is why different shift size needed between od and even size.
    # Given that these two conditions are fulfilled, we are happy and aligned, the way to test it is as follows:
    # The input image, when interpolated (regular bicubic) is exactly aligned with ground truth.

    # First calculate the current center of mass for the kernel
    current_center_of_mass = measurements.center_of_mass(kernel)

    # The second ("+ 0.5 * ....") is for applying condition 2 from the comments above
    wanted_center_of_mass = np.array(kernel.shape) / 2 + 0.5 * (sf - (kernel.shape[0] % 2))

    # Define the shift vector for the kernel shifting (x,y)
    shift_vec = wanted_center_of_mass - current_center_of_mass

    # Before applying the shift, we first pad the kernel so that nothing is lost due to the shift
    # (biggest shift among dims + 1 for safety)
    kernel = np.pad(kernel, np.int(np.ceil(np.max(shift_vec))) + 1, 'constant')

    # Finally shift the kernel and return
    return interpolation.shift(kernel, shift_vec)

# Interpolation kernels
def cubic(x):
    absx = np.abs(x)
    absx2 = absx ** 2
    absx3 = absx ** 3
    return ((1.5*absx3 - 2.5*absx2 + 1) * (absx <= 1) +
            (-0.5*absx3 + 2.5*absx2 - 4*absx + 2) * ((1 < absx) & (absx <= 2)))

def lanczos2(x):
    return (((np.sin(math.pi*x) * np.sin(math.pi*x/2) + np.finfo(np.float32).eps) /
             ((math.pi**2 * x**2 / 2) + np.finfo(np.float32).eps))
            * (abs(x) < 2))

def box(x):
    return ((-0.5 <= x) & (x < 0.5)) * 1.0

def lanczos3(x):
    return (((np.sin(math.pi*x) * np.sin(math.pi*x/3) + np.finfo(np.float32).eps) /
            ((math.pi**2 * x**2 / 3) + np.finfo(np.float32).eps))
            * (abs(x) < 3))

def linear(x):
    return (x + 1) * ((-1 <= x) & (x < 0)) + (1 - x) * ((0 <= x) & (x <= 1))

from types import SimpleNamespace 

#############################
# Training functions (from training.py)
#############################

def train(opt, Gs, Zs, reals, NoiseAmp, real_=None, real=None):
    """
    Train the SinGAN model.
    
    Args:
        opt: Options
        Gs: List of generator networks
        Zs: List of noise tensors
        reals: List of real images at different scales
        NoiseAmp: List of noise amplitudes
        real_: Original real image (optional, will be read if not provided)
        real: Processed real image at scale1 (optional, will be processed if not provided)
    """
    # Only read the image if not provided
    if real_ is None:
        real_ = read_image(opt)
    
    in_s = 0
    scale_num = 0
    
    # Only adjust scales and create pyramid if not already done
    if real is None or len(reals) == 0:
        real = imresize(real_, scale_factor=opt.scale1)
        reals = creat_reals_pyramid(real, reals, opt)
    
    nfc_prev = 0

    while scale_num < opt.stop_scale + 1:
        opt.nfc = min(opt.nfc_init * pow(2, math.floor(scale_num / 4)), 128)
        opt.min_nfc = min(opt.min_nfc_init * pow(2, math.floor(scale_num / 4)), 128)

        opt.out_ = generate_dir2save(opt)
        opt.outf = '%s/%d' % (opt.out_, scale_num)
        try:
            os.makedirs(opt.outf)
        except OSError:
            pass

        plt.imsave('%s/real_scale.png' % (opt.outf), convert_image_np(reals[scale_num]), vmin=0, vmax=1)

        D_curr, G_curr = init_models(opt)
        if (nfc_prev == opt.nfc):
            G_curr.load_state_dict(torch.load('%s/%d/netG.pth' % (opt.out_, scale_num-1), map_location=opt.device))
            D_curr.load_state_dict(torch.load('%s/%d/netD.pth' % (opt.out_, scale_num-1), map_location=opt.device))

        z_curr, in_s, G_curr = train_single_scale(D_curr, G_curr, reals, Gs, Zs, in_s, NoiseAmp, opt)

        G_curr = reset_grads(G_curr, False)
        G_curr.eval()
        D_curr = reset_grads(D_curr, False)
        D_curr.eval()

        Gs.append(G_curr)
        Zs.append(z_curr)
        NoiseAmp.append(opt.noise_amp)

        torch.save(Zs, '%s/Zs.pth' % (opt.out_))
        torch.save(Gs, '%s/Gs.pth' % (opt.out_))
        torch.save(reals, '%s/reals.pth' % (opt.out_))
        torch.save(NoiseAmp, '%s/NoiseAmp.pth' % (opt.out_))

        scale_num += 1
        nfc_prev = opt.nfc
        del D_curr, G_curr
    return

def train_single_scale(netD, netG, reals, Gs, Zs, in_s, NoiseAmp, opt, centers=None):
    # Enable anomaly detection to find the gradient issue
    torch.autograd.set_detect_anomaly(True)

    real = reals[len(Gs)]
    opt.nzx = real.shape[2]  # +(opt.ker_size-1)*(opt.num_layer)
    opt.nzy = real.shape[3]  # +(opt.ker_size-1)*(opt.num_layer)
    opt.receptive_field = opt.ker_size + ((opt.ker_size - 1) * (opt.num_layer - 1)) * opt.stride
    pad_noise = int(((opt.ker_size - 1) * opt.num_layer) / 2)
    pad_image = int(((opt.ker_size - 1) * opt.num_layer) / 2)
    if opt.mode == 'animation_train':
        opt.nzx = real.shape[2] + (opt.ker_size - 1) * (opt.num_layer)
        opt.nzy = real.shape[3] + (opt.ker_size - 1) * (opt.num_layer)
        pad_noise = 0
    m_noise = nn.ZeroPad2d(int(pad_noise))
    m_image = nn.ZeroPad2d(int(pad_image))

    alpha = opt.alpha

    fixed_noise = generate_noise([opt.nc_z, opt.nzx, opt.nzy], device=opt.device)
    z_opt = torch.full(fixed_noise.shape, 0, device=opt.device)
    z_opt = m_noise(z_opt)

    # setup optimizer
    optimizerD = optim.Adam(netD.parameters(), lr=opt.lr_d, betas=(opt.beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=opt.lr_g, betas=(opt.beta1, 0.999))
    schedulerD = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizerD, milestones=[1600], gamma=opt.gamma)
    schedulerG = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizerG, milestones=[1600], gamma=opt.gamma)

    errD2plot = []
    errG2plot = []
    D_real2plot = []
    D_fake2plot = []
    z_opt2plot = []

    for epoch in range(opt.niter):
        if (Gs == []) & (opt.mode != 'SR_train'):
            z_opt = generate_noise([1, opt.nzx, opt.nzy], device=opt.device)
            z_opt = m_noise(z_opt.expand(1, 3, opt.nzx, opt.nzy))
            noise_ = generate_noise([1, opt.nzx, opt.nzy], device=opt.device)
            noise_ = m_noise(noise_.expand(1, 3, opt.nzx, opt.nzy))
        else:
            noise_ = generate_noise([opt.nc_z, opt.nzx, opt.nzy], device=opt.device)
            noise_ = m_noise(noise_)

        ############################
        # (1) Update D network: maximize D(x) + D(G(z))
        ###########################
        for j in range(opt.Dsteps):
            # train with real
            netD.zero_grad()

            output = netD(real).to(opt.device)
            # D_real_map = output.detach()
            errD_real = -output.mean()  # -a
            errD_real.backward(retain_graph=True)
            D_x = -errD_real.item()

            # train with fake
            if (j == 0) & (epoch == 0):
                if (Gs == []) & (opt.mode != 'SR_train'):
                    prev = torch.full([1, opt.nc_z, opt.nzx, opt.nzy], 0, device=opt.device)
                    in_s = prev
                    prev = m_image(prev)
                    z_prev = torch.full([1, opt.nc_z, opt.nzx, opt.nzy], 0, device=opt.device)
                    z_prev = m_noise(z_prev)
                    opt.noise_amp = 1
                elif opt.mode == 'SR_train':
                    z_prev = in_s
                    criterion = nn.MSELoss()
                    RMSE = torch.sqrt(criterion(real, z_prev))
                    opt.noise_amp = opt.noise_amp_init * RMSE
                    z_prev = m_image(z_prev)
                    prev = z_prev
                else:
                    prev = draw_concat(Gs, Zs, reals, NoiseAmp, in_s, 'rand', m_noise, m_image, opt)
                    prev = m_image(prev)
                    z_prev = draw_concat(Gs, Zs, reals, NoiseAmp, in_s, 'rec', m_noise, m_image, opt)
                    criterion = nn.MSELoss()
                    RMSE = torch.sqrt(criterion(real, z_prev))
                    opt.noise_amp = opt.noise_amp_init * RMSE
                    z_prev = m_image(z_prev)
            else:
                prev = draw_concat(Gs, Zs, reals, NoiseAmp, in_s, 'rand', m_noise, m_image, opt)
                prev = m_image(prev)

            if opt.mode == 'paint_train':
                prev = quant2centers(prev, centers)
                plt.imsave('%s/prev.png' % (opt.outf), convert_image_np(prev), vmin=0, vmax=1)

            if (Gs == []) & (opt.mode != 'SR_train'):
                noise = noise_
            else:
                noise = opt.noise_amp * noise_ + prev

            # Create a defensive copy of noise and prev to avoid in-place operations
            noise_copy = noise.clone().detach()
            prev_copy = prev.clone().detach()
            
            fake = netG(noise_copy, prev_copy)
            output = netD(fake.detach())
            errD_fake = output.mean()
            errD_fake.backward(retain_graph=True)
            D_G_z = output.mean().item()

            gradient_penalty = calc_gradient_penalty(netD, real, fake, opt.lambda_grad, opt.device)
            gradient_penalty.backward()

            errD = errD_real + errD_fake + gradient_penalty
            optimizerD.step()

        errD2plot.append(errD.detach())

        ############################
        # (2) Update G network: maximize D(G(z))
        ###########################

        for j in range(opt.Gsteps):
            netG.zero_grad()
            
            # Create defensive copies to avoid in-place operations
            noise_copy = noise.clone().detach()
            prev_copy = prev.clone().detach()
            
            fake = netG(noise_copy, prev_copy)
            output = netD(fake)
            errG = -output.mean()
            errG.backward(retain_graph=True)
            
            if alpha != 0:
                loss = nn.MSELoss()
                if opt.mode == 'paint_train':
                    z_prev = quant2centers(z_prev, centers)
                    plt.imsave('%s/z_prev.png' % (opt.outf), convert_image_np(z_prev), vmin=0, vmax=1)
                
                # Use z_opt (lowercase) consistently
                z_opt_noise = opt.noise_amp * z_opt + z_prev
                rec_loss = alpha * loss(netG(z_opt_noise.detach(), z_prev), real)
                rec_loss.backward(retain_graph=True)
                rec_loss = rec_loss.detach()
            else:
                z_opt_noise = z_opt
                rec_loss = 0

            optimizerG.step()

        errG2plot.append(errG.detach() + rec_loss)
        D_real2plot.append(D_x)
        D_fake2plot.append(D_G_z)
        z_opt2plot.append(rec_loss)

        if epoch % 25 == 0 or epoch == (opt.niter - 1):
            print('scale %d:[%d/%d]' % (len(Gs), epoch, opt.niter))

        if epoch % 500 == 0 or epoch == (opt.niter - 1):
            plt.imsave('%s/fake_sample.png' % (opt.outf), convert_image_np(fake.detach()), vmin=0, vmax=1)
            # Use z_opt_noise (lowercase) for consistency
            plt.imsave('%s/G(z_opt).png' % (opt.outf), convert_image_np(netG(z_opt_noise.detach(), z_prev).detach()), vmin=0, vmax=1)
            torch.save(z_opt, '%s/z_opt.pth' % (opt.outf))

        schedulerD.step()
        schedulerG.step()

    save_networks(netG, netD, z_opt, opt)
    return z_opt, in_s, netG

def draw_concat(Gs, Zs, reals, NoiseAmp, in_s, mode, m_noise, m_image, opt):
    G_z = in_s.clone()  # Make a clone to avoid in-place operations
    if len(Gs) > 0:
        if mode == 'rand':
            count = 0
            pad_noise = int(((opt.ker_size-1)*opt.num_layer)/2)
            if opt.mode == 'animation_train':
                pad_noise = 0
            for G, Z_opt, real_curr, real_next, noise_amp in zip(Gs, Zs, reals, reals[1:], NoiseAmp):
                if count == 0:
                    z = generate_noise([1, Z_opt.shape[2] - 2 * pad_noise, Z_opt.shape[3] - 2 * pad_noise], device=opt.device)
                    z = z.expand(1, 3, z.shape[2], z.shape[3])
                else:
                    z = generate_noise([opt.nc_z, Z_opt.shape[2] - 2 * pad_noise, Z_opt.shape[3] - 2 * pad_noise], device=opt.device)
                z = m_noise(z)
                G_z_curr = G_z[:, :, 0:real_curr.shape[2], 0:real_curr.shape[3]]
                G_z_curr = m_image(G_z_curr)
                z_in = noise_amp * z + G_z_curr  # Non-in-place addition
                G_z = G(z_in.detach(), G_z_curr)
                G_z = imresize(G_z, scale_factor=1/opt.scale_factor)
                G_z = G_z[:, :, 0:real_next.shape[2], 0:real_next.shape[3]]
                count += 1
        if mode == 'rec':
            count = 0
            for G, Z_opt, real_curr, real_next, noise_amp in zip(Gs, Zs, reals, reals[1:], NoiseAmp):
                G_z_curr = G_z[:, :, 0:real_curr.shape[2], 0:real_curr.shape[3]]
                G_z_curr = m_image(G_z_curr)
                z_in = noise_amp * Z_opt + G_z_curr  # Non-in-place addition
                G_z = G(z_in.detach(), G_z_curr)
                G_z = imresize(G_z, scale_factor=1/opt.scale_factor)
                G_z = G_z[:, :, 0:real_next.shape[2], 0:real_next.shape[3]]
                count += 1
    return G_z

def train_paint(opt, Gs, Zs, reals, NoiseAmp, centers, paint_inject_scale):
    in_s = torch.full(reals[0].shape, 0, device=opt.device)
    scale_num = 0
    nfc_prev = 0

    while scale_num < opt.stop_scale + 1:
        if scale_num != paint_inject_scale:
            scale_num += 1
            nfc_prev = opt.nfc
            continue
        else:
            opt.nfc = min(opt.nfc_init * pow(2, math.floor(scale_num / 4)), 128)
            opt.min_nfc = min(opt.min_nfc_init * pow(2, math.floor(scale_num / 4)), 128)

            opt.out_ = generate_dir2save(opt)
            opt.outf = '%s/%d' % (opt.out_, scale_num)
            try:
                os.makedirs(opt.outf)
            except OSError:
                pass

            plt.imsave('%s/in_scale.png' % (opt.outf), convert_image_np(reals[scale_num]), vmin=0, vmax=1)

            D_curr, G_curr = init_models(opt)

            z_curr, in_s, G_curr = train_single_scale(D_curr, G_curr, reals[:scale_num+1], Gs[:scale_num], Zs[:scale_num], in_s, NoiseAmp[:scale_num], opt, centers=centers)

            G_curr = reset_grads(G_curr, False)
            G_curr.eval()
            D_curr = reset_grads(D_curr, False)
            D_curr.eval()

            Gs[scale_num] = G_curr
            Zs[scale_num] = z_curr
            NoiseAmp[scale_num] = opt.noise_amp

            torch.save(Zs, '%s/Zs.pth' % (opt.out_))
            torch.save(Gs, '%s/Gs.pth' % (opt.out_))
            torch.save(reals, '%s/reals.pth' % (opt.out_))
            torch.save(NoiseAmp, '%s/NoiseAmp.pth' % (opt.out_))

            scale_num += 1
            nfc_prev = opt.nfc
        del D_curr, G_curr
    return

def init_models(opt):
    # Generator initialization:
    netG = GeneratorConcatSkip2CleanAdd(opt).to(opt.device)
    netG.apply(weights_init)
    if opt.netG != '':
        netG.load_state_dict(torch.load('%s' % opt.netG, map_location=opt.device))
    
    # Discriminator initialization:
    netD = WDiscriminator(opt).to(opt.device)
    netD.apply(weights_init)
    if opt.netD != '':
        netD.load_state_dict(torch.load('%s' % opt.netD, map_location=opt.device))
    
    # Set models to training mode
    netG.train()
    netD.train()
    
    # Print model architectures
    print(netG)
    print(netD)
    
    return netD, netG

#############################
# Manipulation functions (from manipulate.py)
#############################

def generate_gif(Gs, Zs, reals, NoiseAmp, opt, alpha=0.1, beta=0.9, start_scale=2, fps=10):
    in_s = torch.full(Zs[0].shape, 0, device=opt.device)
    images_cur = []
    count = 0

    for G, Z_opt, noise_amp, real in zip(Gs, Zs, NoiseAmp, reals):
        pad_image = int(((opt.ker_size - 1) * opt.num_layer) / 2)
        nzx = Z_opt.shape[2]
        nzy = Z_opt.shape[3]
        m_image = nn.ZeroPad2d(int(pad_image))
        images_prev = images_cur
        images_cur = []
        if count == 0:
            z_rand = generate_noise([1, nzx, nzy], device=opt.device)
            z_rand = z_rand.expand(1, 3, Z_opt.shape[2], Z_opt.shape[3])
            z_prev1 = 0.95 * Z_opt + 0.05 * z_rand
            z_prev2 = Z_opt
        else:
            z_prev1 = 0.95 * Z_opt + 0.05 * generate_noise([opt.nc_z, nzx, nzy], device=opt.device)
            z_prev2 = Z_opt

        for i in range(0, 100, 1):
            if count == 0:
                z_rand = generate_noise([1, nzx, nzy], device=opt.device)
                z_rand = z_rand.expand(1, 3, Z_opt.shape[2], Z_opt.shape[3])
                diff_curr = beta * (z_prev1 - z_prev2) + (1 - beta) * z_rand
            else:
                diff_curr = beta * (z_prev1 - z_prev2) + (1 - beta) * (generate_noise([opt.nc_z, nzx, nzy], device=opt.device))

            z_curr = alpha * Z_opt + (1 - alpha) * (z_prev1 + diff_curr)
            z_prev2 = z_prev1
            z_prev1 = z_curr

            if images_prev == []:
                I_prev = in_s
            else:
                I_prev = images_prev[i]
                I_prev = imresize(I_prev, scale_factor=1/opt.scale_factor)
                I_prev = I_prev[:, :, 0:real.shape[2], 0:real.shape[3]]
                I_prev = m_image(I_prev)
            if count < start_scale:
                z_curr = Z_opt

            z_in = noise_amp * z_curr + I_prev
            I_curr = G(z_in.detach(), I_prev)

            if (count == len(Gs) - 1):
                I_curr = denorm(I_curr).detach()
                I_curr = I_curr[0, :, :, :].cpu().numpy()
                I_curr = I_curr.transpose(1, 2, 0) * 255
                I_curr = I_curr.astype(np.uint8)

            images_cur.append(I_curr)
        count += 1
    dir2save = generate_dir2save(opt)
    try:
        os.makedirs('%s/start_scale=%d' % (dir2save, start_scale))
    except OSError:
        pass
    imageio.mimsave('%s/start_scale=%d/alpha=%f_beta=%f.gif' % (dir2save, start_scale, alpha, beta), images_cur, fps=fps)
    del images_cur

def SinGAN_generate(Gs, Zs, reals, NoiseAmp, opt, in_s=None, scale_v=1, scale_h=1, n=0, gen_start_scale=0, num_samples=50):
    if in_s is None:
        in_s = torch.full(reals[0].shape, 0, device=opt.device)
    images_cur = []
    for G, Z_opt, noise_amp in zip(Gs, Zs, NoiseAmp):
        pad1 = ((opt.ker_size - 1) * opt.num_layer) / 2
        m = nn.ZeroPad2d(int(pad1))
        nzx = (Z_opt.shape[2] - pad1 * 2) * scale_v
        nzy = (Z_opt.shape[3] - pad1 * 2) * scale_h

        images_prev = images_cur
        images_cur = []

        for i in range(0, num_samples, 1):
            if n == 0:
                z_curr = generate_noise([1, nzx, nzy], device=opt.device)
                z_curr = z_curr.expand(1, 3, z_curr.shape[2], z_curr.shape[3])
                z_curr = m(z_curr)
            else:
                z_curr = generate_noise([opt.nc_z, nzx, nzy], device=opt.device)
                z_curr = m(z_curr)

            if images_prev == []:
                I_prev = m(in_s)
            else:
                I_prev = images_prev[i]
                I_prev = imresize(I_prev, scale_factor=1/opt.scale_factor)
                if opt.mode != "SR":
                    I_prev = I_prev[:, :, 0:round(scale_v * reals[n].shape[2]), 0:round(scale_h * reals[n].shape[3])]
                    I_prev = m(I_prev)
                    I_prev = I_prev[:, :, 0:z_curr.shape[2], 0:z_curr.shape[3]]
                    I_prev = upsampling(I_prev, z_curr.shape[2], z_curr.shape[3])
                else:
                    I_prev = m(I_prev)

            if n < gen_start_scale:
                z_curr = Z_opt

            z_in = noise_amp * (z_curr) + I_prev
            I_curr = G(z_in.detach(), I_prev)

            if n == len(reals) - 1:
                if opt.mode == 'train':
                    dir2save = '%s/RandomSamples/%s/gen_start_scale=%d' % (opt.out, opt.input_name[:-4], gen_start_scale)
                else:
                    dir2save = generate_dir2save(opt)
                try:
                    os.makedirs(dir2save)
                except OSError:
                    pass
                if (opt.mode != "harmonization") & (opt.mode != "editing") & (opt.mode != "SR") & (opt.mode != "paint2image"):
                    plt.imsave('%s/%d.png' % (dir2save, i), convert_image_np(I_curr.detach()), vmin=0, vmax=1)
            images_cur.append(I_curr)
        n += 1
    return I_curr.detach()

#############################
# Main functions for Google Colab
#############################

def get_arguments():
    parser = argparse.ArgumentParser()
    
    # Basic parameters
    parser.add_argument('--mode', help='task to be performed', 
                        default='train', choices=['train', 'random_samples', 'animation', 
                                                'harmonization', 'editing', 'paint2image', 'SR'])
    parser.add_argument('--not_cuda', action='store_true', help='disables cuda', default=False)
    
    # Load, input, save configurations
    parser.add_argument('--netG', default='', help="path to netG (to continue training)")
    parser.add_argument('--netD', default='', help="path to netD (to continue training)")
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    parser.add_argument('--nc_z', type=int, help='noise # channels', default=3)
    parser.add_argument('--nc_im', type=int, help='image # channels', default=3)
    parser.add_argument('--out', help='output folder', default='Output')
        
    # Networks hyper parameters
    parser.add_argument('--nfc', type=int, default=32)
    parser.add_argument('--min_nfc', type=int, default=32)
    parser.add_argument('--ker_size', type=int, help='kernel size', default=3)
    parser.add_argument('--num_layer', type=int, help='number of layers', default=5)
    parser.add_argument('--stride', help='stride', default=1)
    parser.add_argument('--padd_size', type=int, help='net pad size', default=0)
        
    # Pyramid parameters
    parser.add_argument('--scale_factor', type=float, help='pyramid scale factor', default=0.75)
    parser.add_argument('--noise_amp', type=float, help='addative noise cont weight', default=0.1)
    parser.add_argument('--min_size', type=int, help='image minimal size at the coarser scale', default=25)
    parser.add_argument('--max_size', type=int, help='image maximal size at the coarser scale', default=250)

    # Optimization hyper parameters
    parser.add_argument('--niter', type=int, default=2000, help='number of epochs to train per scale')
    parser.add_argument('--gamma', type=float, help='scheduler gamma', default=0.1)
    parser.add_argument('--lr_g', type=float, default=0.0005, help='learning rate, default=0.0005')
    parser.add_argument('--lr_d', type=float, default=0.0005, help='learning rate, default=0.0005')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam, default=0.5')
    parser.add_argument('--Gsteps', type=int, help='Generator inner steps', default=3)
    parser.add_argument('--Dsteps', type=int, help='Discriminator inner steps', default=3)
    parser.add_argument('--lambda_grad', type=float, help='gradient penelty weight', default=0.1)
    parser.add_argument('--alpha', type=float, help='reconstruction loss weight', default=10)

    # Input/Output parameters
    parser.add_argument('--input_dir', help='input image directory', default='Input/Images')
    parser.add_argument('--input_name', help='input image name')
    
    # Random samples parameters
    parser.add_argument('--gen_start_scale', type=int, help='generation start scale', default=0)
    parser.add_argument('--scale_h', type=float, help='horizontal resize factor for random samples', default=1.5)
    parser.add_argument('--scale_v', type=float, help='vertical resize factor for random samples', default=1)
    
    # Animation parameters
    parser.add_argument('--alpha_animation', type=float, help='animation random walk first moment', default=0.1)
    
    # Harmonization/Editing/Paint parameters
    parser.add_argument('--ref_dir', help='input reference directory', default='Input/Harmonization')
    parser.add_argument('--ref_name', help='reference image name')
    parser.add_argument('--harmonization_start_scale', type=int, help='harmonization injection scale', default=0)
    parser.add_argument('--editing_start_scale', type=int, help='editing injection scale', default=0)
    parser.add_argument('--paint_start_scale', type=int, help='paint injection scale', default=0)
    parser.add_argument('--quantization_flag', type=bool, help='specify if to perform color quantization for paint2image', default=False)
    
    # SR parameters
    parser.add_argument('--sr_factor', type=float, help='super resolution factor', default=4)
    
    return parser

def upload_image(target_dir):
    """Function to upload an image in Google Colab"""
    if not IN_COLAB:
        print("This function only works in Google Colab")
        return
    
    uploaded = files.upload()
    
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    
    for filename in uploaded.keys():
        # Save the uploaded image to the target directory
        with open(os.path.join(target_dir, filename), 'wb') as f:
            f.write(uploaded[filename])
        print(f"Uploaded {filename} to {target_dir}")
        return filename  # Return the filename
    return None

def display_images(directory, num_images=10):
    """Function to display images in a directory in Google Colab"""
    image_files = [f for f in os.listdir(directory) if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    if len(image_files) == 0:
        print(f"No images found in {directory}")
        return
    
    num_to_display = min(num_images, len(image_files))
    plt.figure(figsize=(15, 3 * num_to_display))
    
    for i, img_file in enumerate(image_files[:num_to_display]):
        img_path = os.path.join(directory, img_file)
        img = plt.imread(img_path)
        plt.subplot(num_to_display, 1, i+1)
        plt.imshow(img)
        plt.title(img_file)
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def run_training(opt):
    """Function to run the training process"""
    Gs = []
    Zs = []
    reals = []
    NoiseAmp = []
    dir2save = generate_dir2save(opt)

    if os.path.exists(dir2save):
        print('Trained model already exists')
        return None
    else:
        try:
            os.makedirs(dir2save)
        except OSError:
            pass
        
        real = read_image(opt)
        adjust_scales2image(real, opt)
        train(opt, Gs, Zs, reals, NoiseAmp)
        SinGAN_generate(Gs, Zs, reals, NoiseAmp, opt)
        return (Gs, Zs, reals, NoiseAmp)

def run_random_samples(opt):
    """Function to generate random samples"""
    Gs = []
    Zs = []
    reals = []
    NoiseAmp = []
    dir2save = generate_dir2save(opt)
    
    if dir2save is None:
        print('Task does not exist')
        return None
    elif os.path.exists(dir2save):
        print(f'Random samples for image {opt.input_name}, start scale={opt.gen_start_scale}, already exist')
        return None
    else:
        try:
            os.makedirs(dir2save)
        except OSError:
            pass
        
        real = read_image(opt)
        adjust_scales2image(real, opt)
        Gs, Zs, reals, NoiseAmp = load_trained_pyramid(opt)
        in_s = generate_in2coarsest(reals, 1, 1, opt)
        SinGAN_generate(Gs, Zs, reals, NoiseAmp, opt, gen_start_scale=opt.gen_start_scale)
        return dir2save

def run_animation(opt):
    """Function to generate animation"""
    Gs = []
    Zs = []
    reals = []
    NoiseAmp = []
    dir2save = generate_dir2save(opt)
    
    if os.path.exists(dir2save):
        print("Output already exists")
        return None
    else:
        opt.min_size = 20
        opt.mode = 'animation_train'
        real = read_image(opt)
        adjust_scales2image(real, opt)
        dir2trained_model = generate_dir2save(opt)
        
        if os.path.exists(dir2trained_model):
            Gs, Zs, reals, NoiseAmp = load_trained_pyramid(opt)
            opt.mode = 'animation'
        else:
            train(opt, Gs, Zs, reals, NoiseAmp)
            opt.mode = 'animation'
        
        try:
            os.makedirs(dir2save)
        except OSError:
            pass
        
        for start_scale in range(0, 3, 1):
            for b in range(80, 100, 5):
                generate_gif(Gs, Zs, reals, NoiseAmp, opt, beta=b/100, start_scale=start_scale)
        
        return dir2save

def run_harmonization(opt):
    """Function to perform harmonization"""
    Gs = []
    Zs = []
    reals = []
    NoiseAmp = []
    dir2save = generate_dir2save(opt)
    
    if dir2save is None:
        print('Task does not exist')
        return None
    else:
        try:
            os.makedirs(dir2save)
        except OSError:
            pass
        
        real = read_image(opt)
        real = adjust_scales2image(real, opt)
        Gs, Zs, reals, NoiseAmp = load_trained_pyramid(opt)
        
        if (opt.harmonization_start_scale < 1) | (opt.harmonization_start_scale > (len(Gs)-1)):
            print(f"Injection scale should be between 1 and {len(Gs)-1}")
            return None
        else:
            ref = read_image_dir('%s/%s' % (opt.ref_dir, opt.ref_name), opt)
            mask = read_image_dir('%s/%s_mask%s' % (opt.ref_dir, opt.ref_name[:-4], opt.ref_name[-4:]), opt)
            
            if ref.shape[3] != real.shape[3]:
                mask = imresize_to_shape(mask, [real.shape[2], real.shape[3]], opt)
                mask = mask[:, :, :real.shape[2], :real.shape[3]]
                ref = imresize_to_shape(ref, [real.shape[2], real.shape[3]], opt)
                ref = ref[:, :, :real.shape[2], :real.shape[3]]
            
            mask = dilate_mask(mask, opt)

            N = len(reals) - 1
            n = opt.harmonization_start_scale
            in_s = imresize(ref, scale_factor=pow(opt.scale_factor, (N - n + 1)))
            in_s = in_s[:, :, :reals[n - 1].shape[2], :reals[n - 1].shape[3]]
            in_s = imresize(in_s, scale_factor=1/opt.scale_factor)
            in_s = in_s[:, :, :reals[n].shape[2], :reals[n].shape[3]]
            out = SinGAN_generate(Gs[n:], Zs[n:], reals, NoiseAmp[n:], opt, in_s, n=n, num_samples=1)
            out = (1-mask)*real + mask*out
            plt.imsave('%s/start_scale=%d.png' % (dir2save, opt.harmonization_start_scale), convert_image_np(out.detach()), vmin=0, vmax=1)
            
            return dir2save

def run_editing(opt):
    """Function to perform editing"""
    Gs = []
    Zs = []
    reals = []
    NoiseAmp = []
    dir2save = generate_dir2save(opt)
    
    if dir2save is None:
        print('Task does not exist')
        return None
    else:
        try:
            os.makedirs(dir2save)
        except OSError:
            pass
        
        real = read_image(opt)
        real = adjust_scales2image(real, opt)
        Gs, Zs, reals, NoiseAmp = load_trained_pyramid(opt)
        
        if (opt.editing_start_scale < 1) | (opt.editing_start_scale > (len(Gs)-1)):
            print(f"Injection scale should be between 1 and {len(Gs)-1}")
            return None
        else:
            ref = read_image_dir('%s/%s' % (opt.ref_dir, opt.ref_name), opt)
            mask = read_image_dir('%s/%s_mask%s' % (opt.ref_dir, opt.ref_name[:-4], opt.ref_name[-4:]), opt)
            
            if ref.shape[3] != real.shape[3]:
                mask = imresize_to_shape(mask, [real.shape[2], real.shape[3]], opt)
                mask = mask[:, :, :real.shape[2], :real.shape[3]]
                ref = imresize_to_shape(ref, [real.shape[2], real.shape[3]], opt)
                ref = ref[:, :, :real.shape[2], :real.shape[3]]
            
            mask = dilate_mask(mask, opt)

            N = len(reals) - 1
            n = opt.editing_start_scale
            in_s = imresize(ref, scale_factor=pow(opt.scale_factor, (N - n + 1)))
            in_s = in_s[:, :, :reals[n - 1].shape[2], :reals[n - 1].shape[3]]
            in_s = imresize(in_s, scale_factor=1/opt.scale_factor)
            in_s = in_s[:, :, :reals[n].shape[2], :reals[n].shape[3]]
            out = SinGAN_generate(Gs[n:], Zs[n:], reals, NoiseAmp[n:], opt, in_s, n=n, num_samples=1)
            
            plt.imsave('%s/start_scale=%d_masked.png' % (dir2save, opt.editing_start_scale), convert_image_np((1-mask)*real + mask*out), vmin=0, vmax=1)
            plt.imsave('%s/start_scale=%d_unmasked.png' % (dir2save, opt.editing_start_scale), convert_image_np(out), vmin=0, vmax=1)
            
            return dir2save

def run_paint2image(opt):
    """Function to perform paint to image conversion"""
    Gs = []
    Zs = []
    reals = []
    NoiseAmp = []
    dir2save = generate_dir2save(opt)
    
    if dir2save is None:
        print('Task does not exist')
        return None
    else:
        try:
            os.makedirs(dir2save)
        except OSError:
            pass
        
        real = read_image(opt)
        real = adjust_scales2image(real, opt)
        Gs, Zs, reals, NoiseAmp = load_trained_pyramid(opt)
        
        if (opt.paint_start_scale < 1) | (opt.paint_start_scale > (len(Gs)-1)):
            print(f"Injection scale should be between 1 and {len(Gs)-1}")
            return None
        else:
            ref = read_image_dir('%s/%s' % (opt.ref_dir, opt.ref_name), opt)
            
            if ref.shape[3] != real.shape[3]:
                ref = imresize_to_shape(ref, [real.shape[2], real.shape[3]], opt)
                ref = ref[:, :, :real.shape[2], :real.shape[3]]

            N = len(reals) - 1
            n = opt.paint_start_scale
            in_s = imresize(ref, scale_factor=pow(opt.scale_factor, (N - n + 1)))
            in_s = in_s[:, :, :reals[n - 1].shape[2], :reals[n - 1].shape[3]]
            in_s = imresize(in_s, scale_factor=1/opt.scale_factor)
            in_s = in_s[:, :, :reals[n].shape[2], :reals[n].shape[3]]
            
            if opt.quantization_flag:
                opt.mode = 'paint_train'
                dir2trained_model = generate_dir2save(opt)
                
                real_s = imresize(real, scale_factor=pow(opt.scale_factor, (N - n)))
                real_s = real_s[:, :, :reals[n].shape[2], :reals[n].shape[3]]
                real_quant, centers = quant(real_s, opt.device)
                
                plt.imsave('%s/real_quant.png' % dir2save, convert_image_np(real_quant), vmin=0, vmax=1)
                plt.imsave('%s/in_paint.png' % dir2save, convert_image_np(in_s), vmin=0, vmax=1)
                
                in_s = quant2centers(ref, centers)
                in_s = imresize(in_s, scale_factor=pow(opt.scale_factor, (N - n)))
                in_s = in_s[:, :, :reals[n].shape[2], :reals[n].shape[3]]
                
                plt.imsave('%s/in_paint_quant.png' % dir2save, convert_image_np(in_s), vmin=0, vmax=1)
                
                if os.path.exists(dir2trained_model):
                    Gs, Zs, reals, NoiseAmp = load_trained_pyramid(opt)
                    opt.mode = 'paint2image'
                else:
                    train_paint(opt, Gs, Zs, reals, NoiseAmp, centers, opt.paint_start_scale)
                    opt.mode = 'paint2image'
            
            out = SinGAN_generate(Gs[n:], Zs[n:], reals, NoiseAmp[n:], opt, in_s, n=n, num_samples=1)
            plt.imsave('%s/start_scale=%d.png' % (dir2save, opt.paint_start_scale), convert_image_np(out.detach()), vmin=0, vmax=1)
            
            return dir2save

def run_super_resolution(opt):
    """Function to perform super resolution"""
    Gs = []
    Zs = []
    reals = []
    NoiseAmp = []
    dir2save = generate_dir2save(opt)
    
    if dir2save is None:
        print('Task does not exist')
        return None
    else:
        try:
            os.makedirs(dir2save)
        except OSError:
            pass

        mode = opt.mode
        in_scale, iter_num = calc_init_scale(opt)
        opt.scale_factor = 1 / in_scale
        opt.scale_factor_init = 1 / in_scale
        opt.mode = 'train'
        dir2trained_model = generate_dir2save(opt)
        
        if os.path.exists(dir2trained_model):
            Gs, Zs, reals, NoiseAmp = load_trained_pyramid(opt)
            opt.mode = mode
        else:
            print('*** Train SinGAN for SR ***')
            real = read_image(opt)
            opt.min_size = 18
            real = adjust_scales2image_SR(real, opt)
            train(opt, Gs, Zs, reals, NoiseAmp)
            opt.mode = mode
        
        print(f'Super resolution factor: {pow(in_scale, iter_num)}')
        Zs_sr = []
        reals_sr = []
        NoiseAmp_sr = []
        Gs_sr = []
        real = reals[-1]
        real_ = real
        opt.scale_factor = 1 / in_scale
        opt.scale_factor_init = 1 / in_scale
        
        for j in range(1, iter_num + 1, 1):
            real_ = imresize(real_, scale_factor=pow(1/opt.scale_factor, 1))
            reals_sr.append(real_)
            Gs_sr.append(Gs[-1])
            NoiseAmp_sr.append(NoiseAmp[-1])
            z_opt = torch.full(real_.shape, 0, device=opt.device)
            m = nn.ZeroPad2d(5)
            z_opt = m(z_opt)
            Zs_sr.append(z_opt)
        
        out = SinGAN_generate(Gs_sr, Zs_sr, reals_sr, NoiseAmp_sr, opt, in_s=reals_sr[0], num_samples=1)
        out = out[:, :, 0:int(opt.sr_factor * reals[-1].shape[2]), 0:int(opt.sr_factor * reals[-1].shape[3])]
        dir2save = generate_dir2save(opt)
        plt.imsave('%s/%s_HR.png' % (dir2save, opt.input_name[:-4]), convert_image_np(out.detach()), vmin=0, vmax=1)
        
        return dir2save

def colab_main():
    # Parse arguments
    parser = get_arguments()
    opt = parser.parse_args([])  # Start with empty args
    
    print("Starting training...")
    # Set default values for Colab
    opt.not_cuda = not torch.cuda.is_available()  # Automatically determine CUDA availability
    opt.niter = 2000
    opt.min_size = 25
    opt.max_size = 250
    opt.mode = 'train'
    opt.input_dir = 'Input/Images'
    
    # Initialize required values
    opt.scale_factor = 0.75  # Default scale factor
    opt.scale_factor_init = 0.75
    opt.min_nfc = 32
    opt.min_nfc_init = 32
    opt.nfc = 32
    opt.nfc_init = 32
    opt.alpha = 10
    opt.beta1 = 0.5
    opt.nc_im = 3  # RGB image
    opt.nc_z = 3   # Noise dimension
    opt.noise_amp = 0.1
    opt.noise_amp_init = 0.1
    opt.ker_size = 3
    opt.num_layer = 5
    opt.padd_size = 0
    opt.stride = 1
    opt.gamma = 0.1
    opt.lambda_grad = 0.1
    opt.Gsteps = 3
    opt.Dsteps = 3
    opt.lr_g = 0.0005
    opt.lr_d = 0.0005
    opt.out = 'Output'
    opt.manualSeed = None  # Will be set in post_config
    
    # Initialize directories
    os.makedirs(opt.input_dir, exist_ok=True)
    
    if IN_COLAB:
        # Handle image upload in Colab
        uploaded = files.upload()
        for filename in uploaded.keys():
            # Save uploaded image to Input directory
            input_path = os.path.join(opt.input_dir, filename)
            with open(input_path, 'wb') as f:
                f.write(uploaded[filename])
            print(f"Uploaded {filename} to {opt.input_dir}")
            opt.input_name = filename
            break  # Just use the first uploaded image
    else:
        # Use a default image if not in Colab
        test_images = os.listdir(opt.input_dir)
        if test_images:
            opt.input_name = test_images[0]
            print(f"Using existing image: {opt.input_name}")
        else:
            print("No images found in Input/Images directory. Please add an image.")
            return
    
    # Configure the model with post_config
    opt = post_config(opt)
    
    Gs = []
    Zs = []
    reals = []
    NoiseAmp = []
    dir2save = generate_dir2save(opt)
    
    try:
        if os.path.exists(dir2save):
            print("Loading pre-trained model...")
            Gs, Zs, reals, NoiseAmp = load_trained_pyramid(opt)
        else:
            print("Training new model...")
            try:
                # First read the image and adjust scales
                real_ = read_image(opt)
                # This will set opt.scale1 and other scale-related parameters
                real = adjust_scales2image(real_, opt)
                # Now pass the pre-processed images to train
                train(opt, Gs, Zs, reals, NoiseAmp, real_=real_, real=real)
            except RuntimeError as e:
                if "CUDA" in str(e):
                    print("\nError: CUDA (GPU) operation failed. Switching to CPU mode.")
                    opt.not_cuda = True
                    opt.device = torch.device("cpu")
                    # Try again with CPU
                    real_ = read_image(opt)
                    real = adjust_scales2image(real_, opt)
                    train(opt, Gs, Zs, reals, NoiseAmp, real_=real_, real=real)
                else:
                    raise e
        
        # Generate a random sample
        print("Generating random samples...")
        try:
            SinGAN_generate(Gs, Zs, reals, NoiseAmp, opt)
            print("Training and sample generation completed successfully!")
        except Exception as e:
            print(f"Warning: Sample generation failed: {str(e)}")
            print("Training completed successfully, but sample generation failed.")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
        
    return opt, Gs, Zs, reals, NoiseAmp

# Run the main function when the script is executed
if __name__ == "__main__":
    colab_main() 

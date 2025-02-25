import torch
from torch import nn, Tensor, FloatTensor
import torch.nn.functional as F

from typing import Dict, Iterable, Callable

from PIL import Image
import numpy as np
from math import exp
from scipy import optimize

from lucent.optvis import param, transform
from lucent.optvis.objectives import wrap_objective

def makeGaussian(size, fwhm = None, center=None):
    x = np.arange(0, size, 1, float)
    y = x[:,np.newaxis]

    if center is None:
        x0 = y0 = size // 2
    else:
        x0 = center[0]
        y0 = center[1]
    
    return np.exp(-4*np.log(2) * ((x-x0)**2 + (y-y0)**2) / fwhm**2)

class FeatureExtractor(nn.Module):
    def __init__(self, model: nn.Module, layers: Iterable[str]):
        super().__init__()
        self.model = model
        self.layers = layers
        self._features = {layer: torch.empty(0) for layer in layers}

        for layer_id in layers:
            layer = dict([*self.model.named_modules()])[layer_id]
            layer.register_forward_hook(self.save_outputs_hook(layer_id))

    def save_outputs_hook(self, layer_id: str) -> Callable:
        def fn(_, __, output):
            self._features[layer_id] = output
        return fn

    def forward(self, x: Tensor):
        _ = self.model(x)
        return self._features

def fix_parameters(module, value=None):
    """
    Set requires_grad = False for all parameters.
    If a value is passed all parameters are fixed to the value.
    """
    for param in module.parameters():
        if value:
            param.data = FloatTensor(param.data.size()).fill_(value)
        param.requires_grad = False
    return module

def smoothing_laplacian_loss(data, device, weight=1e-3, L=None):
    if L is None:
        L = torch.tensor([[0,-1,0],[-1,-4,-1],[0,-1,0]],device=device)
        
    temp = torch.reshape(data.squeeze(), [data.squeeze().shape[0],
                          np.sqrt(data.squeeze().shape[1]).astype('int'),
                          np.sqrt(data.squeeze().shape[1]).astype('int')])
    temp = torch.square(F.conv2d(temp.unsqueeze(1),L.unsqueeze(0).unsqueeze(0).float(),
                    padding=5))
    return weight * torch.sqrt(torch.sum(temp))

def sparsity_loss(data_1, data_2, weight=1e-3):
    return weight * torch.sum(torch.sum(torch.abs(data_1))) * torch.norm(torch.sum(torch.abs(data_2)))

def smoothing_laplacian_loss_v2(data, device, weight=1e-3, L=None):
    if L is None:
        L = torch.tensor([[0,-1,0],[-1,-4,-1],[0,-1,0]],device=device)
    L = L.unsqueeze(0).unsqueeze(0)
    temp = F.conv2d(data.permute([3,0,1,2]),L.repeat_interleave(data.shape[0],1).float())
    return weight * torch.mean(torch.sum(torch.square(temp),[1,2,3]))

def sparsity_loss_v2(data, weight=1e-3):
    return weight * torch.mean(torch.sum(torch.sqrt(torch.sum(torch.square(data),[0,1])),1))

def l1_loss(data, weight=1e-3):
    return weight * torch.mean(torch.sum(torch.abs(data),[0,1,2]))

def sta(neural_data,img_data,size=20):
    sta = []
    for c in range(neural_data.shape[1]):
        res = []
        for n in range(neural_data.shape[0]):
            temp = (neural_data[n,c])
            res.append((temp*torch.mean(F.interpolate(img_data[n].unsqueeze(0),
                                                      size=[size,size]).squeeze(),0)).detach().numpy())
        res = np.asarray(res)
        res = np.sum(res,0)
        res = (res + np.abs(res.min()))
        sta.append(res/res.max())
    return np.asarray(sta)

def load_sta(sta,mod_shape,device):
    out = []
    for i in range(len(sta)):
        temp = torch.tensor(sta[i]).to(device)
        temp = F.interpolate(temp.unsqueeze(0).unsqueeze(0), size=[mod_shape[2],mod_shape[3]]).squeeze()
        temp = temp.unsqueeze(0)
        out.append(temp.repeat_interleave(mod_shape[1],0))
    out = torch.stack(out)
    return out.permute([1,2,3,0])

def load(path,sz):
    return np.array(Image.open(path).resize((sz,sz))) / 255

def load_crop(path,sz):
    im = Image.open(path)
    width, height = im.size   # Get dimensions
    min_size = np.min([width,height])
    left = (width - min_size)/2
    top = (height - min_size)/2
    right = (width + min_size)/2
    bottom = (height + min_size)/2
    # Crop the center of the image
    im = im.crop((left, top, right, bottom))
    return np.array(im.resize((sz,sz))) / 255

def mean_L1(a, b):
    return torch.abs(a-b).mean()

@wrap_objective()
def activation_difference(layer_names, activation_loss_f=mean_L1, transform_f=None):
    def inner(T):
        # first we collect the (constant) activations of image we're computing the difference to
        image_activations = [T(layer_name)[1] for layer_name in layer_names]
        if transform_f is not None:
            image_activations = [transform_f(act) for act in image_activations]

        # we also set get the activations of the optimized image which will change during optimization
        optimization_activations = [T(layer)[0] for layer in layer_names]
        if transform_f is not None:
            optimization_activations = [transform_f(act) for act in optimization_activations]

        # we use the supplied loss function to compute the actual losses
        losses = [activation_loss_f(a, b) for a, b in zip(image_activations, optimization_activations)]
        return sum(losses)

    return inner

def gram_matrix(features, normalize=True):
    C, H, W = features.shape
    features = features.view(C, -1)
    gram = torch.matmul(features, torch.transpose(features, 0, 1))
    if normalize:
        gram = gram / (H * W)
    return gram

def gaussian(height, center_x, center_y, width_x, width_y):
    """Returns a gaussian function with the given parameters"""
    width_x = float(width_x)
    width_y = float(width_y)
    return lambda x,y: height*np.exp(
                -(((center_x-x)/width_x)**2+((center_y-y)/width_y)**2)/2)

def moments(data):
    """Returns (height, x, y, width_x, width_y)
    the gaussian parameters of a 2D distribution by calculating its
    moments """
    total = data.sum()
    X, Y = np.indices(data.shape)
    x = (X*data).sum()/total
    y = (Y*data).sum()/total
    col = data[:, int(y)]
    width_x = np.sqrt(np.abs((np.arange(col.size)-x)**2*col).sum()/col.sum())
    row = data[int(x), :]
    width_y = np.sqrt(np.abs((np.arange(row.size)-y)**2*row).sum()/row.sum())
    height = data.max()
    return height, x, y, width_x, width_y

def fitgaussian(data):
    """Returns (height, x, y, width_x, width_y)
    the gaussian parameters of a 2D distribution found by a fit"""
    params = moments(data)
    errorfunction = lambda p: np.ravel(gaussian(*p)(*np.indices(data.shape)) -
                                 data)
    p, success = optimize.leastsq(errorfunction, params)
    return p

def gabor_patch(shape, pos_yx = None, radius = None, wavelength = None, orientation = None, 
                phase = None, min_brightness = 0, max_brightness = 1):
    
    if pos_yx is None:
        pos_yx = (shape[0] / 2, shape[1] / 2)
    
    assert len(shape) >= 2
    H = shape[len(shape) - 2]
    W = shape[len(shape) - 1]

    assert len(pos_yx) == 2

    # "Bounding box"
    xmax = W
    ymax = H
    xmin = 0
    ymin = 0

    (x, y) = np.meshgrid(np.arange(xmin, xmax), np.arange(ymin, ymax))

    # Rotation (around center of image)
    # The value along the x of the grating is constant
    y_theta = (x * np.sin(orientation) +
               y * np.cos(orientation))

    d = np.sqrt((y - pos_yx[0]) ** 2 + (x - pos_yx[1]) ** 2) - radius
    d[d < - 0.5] = - 0.5
    d[d > 0.5] = 0.5
    envelope = 0.5 - d

    # initially make gratings vary from 0 to 1
    gratings = (np.cos(2 * np.pi / wavelength * y_theta + phase) + 1) / 2
    # make gratings between min_brightness and max_brightness
    gratings = gratings * (max_brightness - min_brightness) + min_brightness
    gb = np.multiply(
        envelope,
        gratings)
    # set the background color
    gb += (1-envelope) * ((max_brightness - min_brightness) / 2  + min_brightness)

    gb = np.reshape(gb, [1, 1, H, W])
    gb = np.repeat(gb, 3, 1)
    return gb

class occlude_pic(object):
    
    def __init__(self, occluder, mask, device = 'cpu'):
        self.occluder_t = torch.tensor(np.transpose(occluder[0], [2, 0, 1])).float().to(device)
        self.mask_t = torch.tensor(np.transpose(mask, [2, 0, 1])).float().to(device)
    
    def __call__(self,img): 
        occluded_input = img[0] * (1-self.mask_t) + self.occluder_t * self.mask_t
        return occluded_input[None,:,:,:]


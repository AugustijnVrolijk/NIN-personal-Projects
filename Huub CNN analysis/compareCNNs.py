import torch
import torch.nn.functional as F
from torch import Tensor, nn, FloatTensor
import os
import pickle
import numpy as np
import time
from torchinfo import summary
from lucent.modelzoo import inceptionv1
from typing import Iterable, Callable
import inspect
import matplotlib.pyplot as plt
import math

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

class FeatureExtractor(nn.Module):
    def __init__(self, model: nn.Module, layers: Iterable[str]):
        super().__init__()
        self.model = model
        self.layers = layers
        self._features = {layer: torch.empty(0) for layer in layers}

        for layer_id in layers:
            self.layer = dict([*self.model.named_modules()])[layer_id]
            self.layer.register_forward_hook(self.save_outputs_hook(layer_id))

    def save_outputs_hook(self, layer_id: str) -> Callable:
        def fn(_, __, output):
            self._features[layer_id] = output
        return fn

    def forward(self, x: Tensor):
        _ = self.model(x)
        return self._features

def makeGaussian(size, fwhm = None, center=None):
    x = np.arange(0, size, 1, float)
    y = x[:,np.newaxis]

    if center is None:
        x0 = y0 = size // 2
    else:
        x0 = center[0]
        y0 = center[1]
    
    return np.exp(-4*np.log(2) * ((x-x0)**2 + (y-y0)**2) / fwhm**2)

class Model(nn.Module):
    """
    Model of neural responses
    """

    def __init__(self,pretrained_model,layer,n_neurons,device=None,debug=False):
        super(Model, self).__init__()
        self.layer = layer
        self.debug = debug
        self.ann = fix_parameters(pretrained_model)
        self.n_neurons = n_neurons
        self.inc_features = FeatureExtractor(self.ann, layers=[self.layer])
        dummy_input = torch.ones(1, 3, 224, 224)
        self.dummy_feats = self.inc_features(dummy_input)
        self.mod_shape = self.dummy_feats[self.layer].shape
        if self.debug:
            self.w_s = torch.nn.Parameter(torch.randn(self.n_neurons, 1, self.mod_shape[-1]*self.mod_shape[-1], 1, 
                                                      device=device))
        else:
            self.w_s = torch.nn.Parameter(torch.randn(self.n_neurons, 1, self.mod_shape[-1]*self.mod_shape[-1], 1, 
                                                      device=device,requires_grad=True))
        self.w_f = torch.nn.Parameter(torch.randn(1, self.n_neurons, 1, self.mod_shape[1], 
                                                  device=device, requires_grad=True))
        self.ann_bn = torch.nn.BatchNorm2d(self.mod_shape[1],momentum=0.9,eps=1e-4,affine=False)
        self.output = torch.nn.Identity()

    def forward(self,x):
        x = self.inc_features(x)
        x = x[self.layer]
        print(f"1 has shape: {x.size()}")
        #torch.Size([1, 480, 28, 28])
        x = F.relu(self.ann_bn(x))
        print(f"2 has shape: {x.size()}")
        #torch.Size([1, 480, 28, 28])
        x = x.view(x.shape[0],x.shape[1],x.shape[2]*x.shape[3],1)
        print(f"3 has shape: {x.size()}")
        #torch.Size([1, 480, 784, 1])
        x = x.permute(0,-1,2,1)
        print(f"4 has shape: {x.size()}")
        #torch.Size([1, 1, 784, 480])
        
        print(f"w_s has shape: {self.w_s.shape}")
        #w_s has shape torch.Size([52, 1, 784, 1])

        x = F.conv2d(x,torch.abs(self.w_s))
        print(f"5 has shape: {x.size()}")
        #torch.Size([1, 52, 1, 480])

        print(f"w_f has shape: {self.w_f.shape}")
        #w_f torch.Size([1, 52, 1, 480])
        x = torch.mul(x,self.w_f)
        print(f"6 has shape: {x.size()}")
        #torch.Size([1, 52, 1, 480])
        x = torch.sum(x,-1,keepdim=True)
        print(f"7 has shape: {x.size()}")
        #torch.Size([1, 52, 1, 1])
        return self.output(x)
    
    def initialize(self):
        nn.init.xavier_normal_(self.w_f)
        if self.debug:
            temp = np.ndarray.flatten(makeGaussian(self.mod_shape[-1], fwhm = self.mod_shape[-1]/20, 
                                                   center=[self.mod_shape[-1]*.3,self.mod_shape[-1]*.7]))
            for i in range(len(self.w_s)):
                self.w_s[i,0,:,0] = torch.tensor(temp)
        else:
            nn.init.xavier_normal_(self.w_s)

def printPickleFile(pickleF:dict, data_filename:str):

    keys = pickleF.keys()
    print("-------------------------------------\nfile:\n{}\n".format(data_filename))
    print(keys)
    for key in keys:
        tempVal = pickleF[key]
        print("{}, {}".format(key, type(tempVal)))
        if isinstance(tempVal, np.ndarray):
            print("shape = {}".format(tempVal.shape))
        elif isinstance(tempVal, torch.Tensor):
            print("shape = {}".format(tempVal.size()))
        elif isinstance(tempVal, list):
            print("shape = {}".format(len(tempVal)))
        else:
            print("not a np array or tensor")

def load_mouse_model(gen_path:str,group_name:str,condition:str, verbose:bool = False):
    data_filename = os.path.join(gen_path, 'snapshots/grid_search_'+ group_name + condition + '.pkl')
    f = open(data_filename,"rb")
    cc = pickle.load(f)
    if verbose:
        printPickleFile(cc, data_filename)
    val_corrs = cc['val_corrs']
    params = cc['params']
    val_corrs = np.array(val_corrs)
    layer = params[np.where(val_corrs==np.nanmax(val_corrs))[0][0].astype('int')][0]
    data_filename = os.path.join(gen_path,'data_NPC_'+ group_name + condition +'.pkl')
    f = open(data_filename,"rb")
    cc = pickle.load(f)
    if verbose:
        printPickleFile(cc, data_filename)

    val_data = cc['val_data']
    good_neurons = cc['good_neurons']
    reds = cc['reds']
    snr = cc['snr']
    mean_reliab = cc['mean_reliab']
    n_neurons = val_data.shape[1]

    pretrained_model = inceptionv1(pretrained=True)
    mouse_model = Model(pretrained_model,layer,n_neurons,device='cpu')
    
    if verbose:
        summarisePTModel(mouse_model)
    preTrainedWeights = torch.load(gen_path + '\\' + group_name + condition + '_' + layer + '_neural_model.pt',map_location=torch.device('cpu'))

    mouse_model.load_state_dict(preTrainedWeights)
    return mouse_model,n_neurons,good_neurons,reds,mean_reliab,snr

def summarisePTModel(mouse_model:Model):
    print(f"{type(mouse_model)} \n")
    attributes = vars(mouse_model)
    for key, val in attributes.items():
        if isinstance(val, torch.nn.Parameter):
            print(f"KEY: {key}\nVALTYPE: {type(val)}\nVAL: {val.size()}\n")

        elif isinstance(val, dict):
            print(f"KEY: {key}\nVALTYPE: {type(val)}\n")
            for subKey, subVal in val.items():
                if isinstance(subVal, FeatureExtractor):
                    print(f"--------------------------------------------------------------\n KEY: {subKey}\nVALTYPE: {type(subVal)}\n")
                    summarisePTModel(subVal)
                    print(f"--------------------------------------------------------------")

                elif isinstance(subVal, Tensor):
                    print(f"        SUBKEY: {subKey}\n        SUBVALTYPE: {type(subVal)}\n      SUBVAL: {subVal.size()}\n")
                else:
                    print(f"        SUBKEY: {subKey}\n        SUBVALTYPE: {type(subVal)}\n      SUBVAL: {subVal}\n")

        else:
            print(f"KEY: {key}\nVALTYPE: {type(val)}\nVAL: {val}\n")
    print(inspect.getsource(mouse_model.forward))

    #specific for the current model analysis
    
    return

def fetchModel(gen_path, group_name, condition, verbose:bool = False):
    
    mouse_model,n_neurons,good_neurons,reds,mean_reliab,snr = load_mouse_model(gen_path, group_name, condition)

    if verbose:
        summarisePTModel(mouse_model)
        dummy_input = torch.ones(1, 3, 224, 224)
        print("here we go\n")
        mouse_model(dummy_input)
    
    return mouse_model

def analyseWeights(weight_tensor, verbose = False):
    print(weight_tensor.shape)
    #detach the gradient, so its just a tensor of numbers
    #convert the tensor to a numpy array
    #squeeze out the excess dimensions of size 1
    xVals = weight_tensor.detach().numpy().squeeze() 
    nNeurons, _ = xVals.shape
    print(xVals.shape)
    minBound, maxBound = -0.5, 0.5

    bins = np.arange(minBound, maxBound, 0.01)
    bin_width = np.diff(bins)
    bin_coord = bins[:-1] + (bin_width/2)
    nBins = bin_coord.shape[0]
    histograms = np.full((nNeurons, nBins), np.nan, dtype=np.float64)
    histogramDense = np.full((nNeurons, nBins), 0, dtype=np.float64)

    for n in range(nNeurons):
        minVal, maxVal = np.min(xVals[n, :]), np.max(xVals[n, :])
        if minVal < minBound or maxVal > maxBound:
            print("bound error")
            exit()

        histograms[n, :] = np.histogram(xVals[n,:],bins=bins)[0] #comes as a tuple, with [0] being the vals and [1] being the bin coords (can be decided automatically)
        binID = np.digitize(xVals[n,:],bins=bins)
        np.add.at(histogramDense[n, :], binID, np.absolute(xVals[n, :]))
       
        try:
            assert np.sum(histogramDense[n, :]) == np.sum(xVals[n, :])
        except AssertionError as e:
            print(np.sum(histogramDense[n, :]))
            print(np.sum(xVals[n, :]))
        
        if verbose:
            plt.bar(bin_coord, histograms[n, :], width=bin_width)
            plt.show()
            plt.bar(bin_coord,histogramDense[n, :], width=bin_width) 
            plt.show()

    meanHisto = np.mean(histograms, axis=0)
    sdHisto = np.std(histograms, axis=0)
    meanHistoDense = np.mean(histogramDense, axis=0)
    sdHistoDense = np.std(histogramDense, axis=0)
    print(meanHisto.shape)
    print(meanHistoDense.shape)

    
    plt.bar(bin_coord,meanHisto, width=bin_width)
    plt.show()
    plt.bar(bin_coord,meanHistoDense, width=bin_width) 
    plt.show()
  
    return

def weightMatrix(weight_tensor, verbose=False):
    print(weight_tensor.shape)
    #detach the gradient, so its just a tensor of numbers
    #convert the tensor to a numpy array
    #squeeze out the excess dimensions of size 1
    xVals = weight_tensor.detach().numpy().squeeze() 
    nNeurons, imgLen = xVals.shape
    print(xVals.shape)
    print(nNeurons, imgLen)
    print(xVals)

    size = math.sqrt(imgLen)
    if not size.is_integer():
        ValueError(f"Img len: {imgLen} cannot be square rooted: {size}")
    size = int(size)

    reshapedXVals = xVals.reshape(nNeurons, size, size)
    meanXVals = np.mean(reshapedXVals, axis=0)
    stdXVals = np.std(reshapedXVals, axis=0)
    print(meanXVals.shape)

    if verbose:
        for n in range(nNeurons):
            plt.imshow(reshapedXVals[n, :, :], cmap='bwr')
            plt.colorbar()        
            plt.show()
            pass
    
    plt.imshow(meanXVals, cmap='bwr')
    plt.colorbar() 
    plt.show()
    return

def main():
    gen_path = r"C:\Users\augus\NIN Stuff\data\Huub data"
    group_name = "Huub_" 
    condition = "NaturalImages_darkReared_VISa_baseline"
    mouse_model = fetchModel(gen_path, group_name, condition)
    first_weight_tensor = mouse_model.w_s #shape nNeurons, 1, img, 1
    second_weight_tensor = mouse_model.w_f #shape 1, nNeurons, 1, features
    analyseWeights(second_weight_tensor)
    breakpoint()
    weightMatrix(first_weight_tensor, True)
    return 

if __name__ == "__main__":
    main()
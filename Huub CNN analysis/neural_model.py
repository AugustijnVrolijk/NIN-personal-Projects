import torch
from torch import nn, Tensor
import torch.nn.functional as F
from typing import Dict, Iterable, Callable
import numpy as np
from helper import makeGaussian, FeatureExtractor, fix_parameters, load_sta

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
        dummy_feats = self.inc_features(dummy_input)
        self.mod_shape = dummy_feats[self.layer].shape
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
        x = F.relu(self.ann_bn(x))
        x = x.view(x.shape[0],x.shape[1],x.shape[2]*x.shape[3],1)
        x = x.permute(0,-1,2,1)
        x = F.conv2d(x,torch.abs(self.w_s))
        x = torch.mul(x,self.w_f)
        x = torch.sum(x,-1,keepdim=True)
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
            
            
            
class ModelsDiff(nn.Module):
    """
    Difference of chronically matched models
    """
    
    def __init__(self,model_0,model_1,chronic_goods):
        super(ModelsDiff, self).__init__()
        self.chronic_goods = chronic_goods
        self.n_neurons = len(self.chronic_goods[0])
        self.model_0 = fix_parameters(model_0)
        self.model_1 = fix_parameters(model_1)
        self.features_0 = FeatureExtractor(self.model_0, layers=["output"])
        self.features_1 = FeatureExtractor(self.model_1, layers=["output"])
        self.output = torch.nn.Identity()

    def forward(self,x):
        x = self.features_1(x)['output'][:,self.chronic_goods[1]] - self.features_0(x)['output'][:,self.chronic_goods[0]]
        return self.output(x)

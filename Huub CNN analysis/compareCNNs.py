import torch
import torch.nn.functional as F
from torch import Tensor, nn, FloatTensor
import os
import pickle
import numpy as np
from torchinfo import summary
from lucent.modelzoo import inceptionv1
from typing import Iterable, Callable

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
            layer = dict([*self.model.named_modules()])[layer_id]
            layer.register_forward_hook(self.save_outputs_hook(layer_id))

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

def load_mouse_model(gen_path,group_name,condition):
    data_filename = os.path.join(gen_path, 'snapshots/grid_search_'+ group_name + condition + '.pkl')
    f = open(data_filename,"rb")
    cc = pickle.load(f)
    print(cc)
    breakpoint()
    val_corrs = cc['val_corrs']
    params = cc['params']
    val_corrs = np.array(val_corrs)
    layer = params[np.where(val_corrs==np.nanmax(val_corrs))[0][0].astype('int')][0]
    data_filename = os.path.join(gen_path,'data_NPC_'+ group_name + condition +'.pkl')
    f = open(data_filename,"rb")
    cc = pickle.load(f)
    val_data = cc['val_data']
    good_neurons = cc['good_neurons']
    reds = cc['reds']
    snr = cc['snr']
    mean_reliab = cc['mean_reliab']
    
    n_neurons = val_data.shape[1]

    pretrained_model = inceptionv1(pretrained=True)
    mouse_model = Model(pretrained_model,layer,n_neurons,device='cpu')
    mouse_model.load_state_dict(torch.load(gen_path + '\\' + group_name + condition + '_' + layer + '_neural_model.pt',map_location=torch.device('cpu')))
    return mouse_model,n_neurons,good_neurons,reds,mean_reliab,snr


def fetchModel():
    gen_path = r"C:\Users\augus\NIN Stuff\data\Huub data"
    group_name = "Huub_" 
    condition = "NaturalImages_darkReared_VISa_baseline"
    mouse_model,n_neurons,good_neurons,reds,mean_reliab,snr = load_mouse_model(gen_path, group_name, condition)
    print("{}\n".format(n_neurons))
    print("{}\n".format(good_neurons))
    print("{}\n".format(reds))
    print("{}\n".format(mean_reliab))
    print("{}\n".format(snr))
    return



def main():
    #modelPath = r"C:\Users\augus\NIN Stuff\data\Huub data\Huub_NaturalImages_darkReared_VISa_baseline_conv2d1_neural_model.pt"
    fetchModel()
   
    return 

if __name__ == "__main__":
    main()
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
from lucent.util import set_seed
from torch.utils.data import Dataset, DataLoader


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
        print(f"output shape: {self.mod_shape}")
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
        #torch.Size([1, 480, 28, 28])
        x = F.relu(self.ann_bn(x))
        #torch.Size([1, 480, 28, 28])
        x = x.view(x.shape[0],x.shape[1],x.shape[2]*x.shape[3],1)
        #torch.Size([1, 480, 784, 1])
        x = x.permute(0,-1,2,1)
        #torch.Size([1, 1, 784, 480])
        
        #w_s has shape torch.Size([52, 1, 784, 1])

        x = F.conv2d(x,torch.abs(self.w_s))
        #torch.Size([1, 52, 1, 480])

        #w_f torch.Size([1, 52, 1, 480])
        x = torch.mul(x,self.w_f)
        #torch.Size([1, 52, 1, 480])
        x = torch.sum(x,-1,keepdim=True)
        #torch.Size([1, 52, 1, 1])
        x = x.squeeze()
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

class dynamicModel(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

class criterionClass():
    def __init__(self, model:Model, smooth_weight, sparse_weight):
        self.MSELoss = torch.nn.MSELoss()
        self.smooth_weight = smooth_weight
        self.sparse_weight = sparse_weight
        self.lapLoss = self.calcSmoothingLapLoss(model.w_s, model.w_f)

    def smoothing_laplacian_loss(self, data, device, weight=1e-3, L=None):
        if L is None:
            L = torch.tensor([[0,-1,0],[-1,-4,-1],[0,-1,0]],device=device)
            
        temp = torch.reshape(data.squeeze(), [data.squeeze().shape[0],
                            np.sqrt(data.squeeze().shape[1]).astype('int'),
                            np.sqrt(data.squeeze().shape[1]).astype('int')])
        temp = torch.square(F.conv2d(temp.unsqueeze(1),L.unsqueeze(0).unsqueeze(0).float(),
                        padding=5))
        return weight * torch.sqrt(torch.sum(temp))

    def calcSmoothingLapLoss(self, w_s, w_f):
        return self.smoothing_laplacian_loss(w_s, torch.device("cuda:0"), weight=self.smooth_weight) + self.sparse_weight * torch.norm(w_f,1)

    def __call__(self, outputs, target):
        return self.MSELoss(outputs, target.float()) + self.lapLoss
    
    def updateParams(self, model:Model):
        self.lapLoss = self.calcSmoothingLapLoss(model.w_s, model.w_f)

class imageData(Dataset):
    def __init__(self, input, target):
        self.input = torch.tensor(input).cuda()
        self.target = torch.tensor(target).cuda()

    def __len__(self):
        return len(self.input)
        
    def __getitem__(self, idx):
        return self.input[idx].clone().detach().requires_grad_(True), self.target[idx].clone().detach().requires_grad_(True)

def trainModel(model:Model, epochs, train_dataloader, val_dataloader, batchSize, optimizer, scheduler, criterion:criterionClass):

    trainingEpoch_loss = []
    validationEpoch_loss = []

    for epoch in range(epochs):
        model.train()
        criterion.updateParams(model)

        train_features, train_labels = next(iter(train_dataloader))
        # Clear the gradients
        optimizer.zero_grad()
        # Forward Pass
        outputs = model(train_features)
        # Find the Loss
        training_loss = criterion(outputs, train_labels)
        # Calculate gradients
        training_loss.backward()
        # Update Weights
        optimizer.step()

        # Calculate Loss
        
        trainingEpoch_loss.append(training_loss.item())
       
       
        model.eval()     # Optional when not using Model Specific layer
        val_features, val_labels = next(iter(val_dataloader))
        validationStep_loss = []
        # Forward Pass
        outputs = model(val_features)
        # Find the Loss
        validation_loss = criterion(outputs, val_labels)
        # Calculate Loss
        validationEpoch_loss.append(validation_loss.item())

        scheduler.step()

        print (f'Epoch [{epoch+1}/{epochs}], train Loss: {training_loss.item():.4f}, val Loss: {validation_loss.item():.4f}')
        
        param_group = optimizer.param_groups
        print(f"Learning Rate: {param_group[0]['lr']}, Weight Decay: {param_group[0]['weight_decay']}")

    return trainingEpoch_loss, validationEpoch_loss

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

def getModel(layer, n_neurons, batching = False, seed = 0, verbose = False):
    
    set_seed(seed)
    pretrained_model = inceptionv1(pretrained=True)
    net = Model(pretrained_model,layer,n_neurons,torch.device("cuda:0"))
    net.initialize()

    if batching:
        net = torch.nn.DataParallel(net)

    net = net.cuda()

    if verbose:
        print('Initialized using Xavier')
        print("Training on {} GPU's".format(torch.cuda.device_count()))
    return net

def getModelHeuristics(grid_filename, verbose=False):
    
    gridfile = open(grid_filename,"rb")
    gridpickleFile = pickle.load(gridfile)
    if verbose:
        printPickleFile(gridpickleFile, grid_filename)

    bestCorrID = np.where(gridpickleFile["val_corrs"]==np.nanmax(gridpickleFile["val_corrs"]))[0][0].astype('int')

    #gridpickleFile["params"][i] format = [layer, learning_rate, smooth_weight, sparse_weight, weight_decay]
    return gridpickleFile["params"][bestCorrID]

def calcLRDecay(nEpochs, decayStep = 1):

    """
    original code:

    nEpochs = 600
    save_epochs = 100
    lr_decay_gamma = 1/3
    lr_decay_step = 3*save_epochs

    a.k.a decays by 1/3 every 300 epochs, in practice only once as we don't train any more after the second decay step at 600
    """
    net_decay = 1/4
    nIter = nEpochs/decayStep

    # Calculate x using the formula derived
    decay_gamma = (net_decay ** (1/nIter))

    return decay_gamma, decayStep

def getData(data_filename, batchSize, validationSize, verbose=False):
    datafile = open(data_filename,"rb")
    datapickleFile = pickle.load(datafile)
    if verbose:
        printPickleFile(datapickleFile, data_filename)

    n_neurons = datapickleFile["mean_reliab"].shape[1]
    images = np.concatenate((datapickleFile["img_data"],datapickleFile["val_img_data"]), axis=0)
    responses = np.concatenate((datapickleFile["train_data"], datapickleFile["val_data"]), axis=0)

    nImgs = images.shape[0]
    val_dataset_size = int(round(validationSize*nImgs))

    indexes = np.arange(nImgs)
    mask = np.random.choice(indexes, nImgs, replace=False)
    
    val_indexes = mask[:val_dataset_size]
    train_indexes = mask[val_dataset_size:]

    if batchSize > val_dataset_size:
        ValueError("batch size is larger than number of images in validation dataset")

    train_dataset = imageData(images[val_indexes].copy(), responses[val_indexes].copy())
    val_dataset = imageData(images[train_indexes].copy(), responses[train_indexes].copy())
    train_dataLoader = DataLoader(train_dataset, batch_size=batchSize, shuffle=True)
    val_dataLoader = DataLoader(val_dataset, batch_size=batchSize, shuffle=True)
    datafile.close()
    return n_neurons, train_dataLoader, val_dataLoader

"""
class ImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
"""

def main():
    
    gen_path = r"C:\Users\augus\NIN Stuff\data\Huub data"
    group_name = "Huub_" 
    condition = "NaturalImages_darkReared_VISa_baseline"
    nEpochs = 4000
    batchSize = 32
    validationSize = 0.2
    seed = 0

    grid_filename = os.path.join(gen_path, 'snapshots/grid_search_'+ group_name + condition + '.pkl')
    data_filename = os.path.join(gen_path,'data_NPC_'+ group_name + condition + '.pkl')

    _, learning_rate, smooth_weight, sparse_weight, weight_decay = getModelHeuristics(grid_filename)
    layer = "mixed5b"
    n_neurons, train_dataLoader, val_dataLoader = getData(data_filename, batchSize, validationSize)
    lr_decay_gamma, lr_decay_step = calcLRDecay(nEpochs)
    print(lr_decay_gamma, lr_decay_step)

    model = getModel(layer, n_neurons, False, seed)
    
    # optimizer and lr scheduler
    criterion = criterionClass(model, smooth_weight, sparse_weight)
    optimizer = torch.optim.Adam(
        [model.w_s,model.w_f],
        lr=learning_rate,
        weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=lr_decay_step,
        gamma=lr_decay_gamma)
    
    trainingEpoch_loss, validationEpoch_loss = trainModel(model, nEpochs, train_dataLoader, val_dataLoader, batchSize, optimizer, scheduler, criterion)

    output = {"trainingEpoch_loss": trainingEpoch_loss, "validationEpoch_loss": validationEpoch_loss}
    output_filename = os.path.join(gen_path,f'Test_cross_validation_scores_{layer}')  
    outputFile = open(output_filename,"wb")
    pickle.dump(output,outputFile)
    outputFile.close()
    return
    
if __name__ == "__main__":
    main()
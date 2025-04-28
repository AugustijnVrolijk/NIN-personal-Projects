### Functions for data processing, model fitting, plotting, storing and fitting MEIs
import torch
import torchvision.models as models 
import torchvision.transforms.functional as F
import torch.nn.functional as nnf
from torchvision import datasets, transforms
from torchvision.transforms.functional import crop
from torch import sigmoid

from lucent.optvis import render, param, transform, objectives
from lucent.modelzoo import inceptionv1, util, inceptionv1_avgPool
from lucent.util import set_seed

import numpy as np
import pickle
from scipy import stats
from scipy.io import savemat,loadmat
import mat73
from pathlib import Path
import os
from os import listdir
from os.path import isfile, join
import h5py
import time

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import shapely.affinity
import shapely.geometry
from skimage.draw import polygon
from skimage.morphology import convex_hull_image
from skimage.filters import gaussian as smoothing
from PIL import Image  

import math
from tqdm import tqdm

from helper import smoothing_laplacian_loss, sparsity_loss, gaussian, moments, fitgaussian, load, gabor_patch, occlude_pic
from neural_model import Model


### Data processing + model fitting:

def format_data(proc,prefix,date,out_path,gen_path,root_dir,subdirs,subsubdirs,areaNames):
    group_names = []
    dates = []
    cond_names = []
    # Loop through areas, subdirectories, and sub-subdirectories
    for area_idx, area in enumerate(areaNames):
        for subdir_idx, subdir in enumerate(subdirs):
            for subsubdir_idx, subsubdir in enumerate(subsubdirs):
                filename_out = fr"{prefix}_{subdir[:-1]}_{area}_{subsubdir[:-1]}_{date}"
                filename_all = os.path.join(out_path,filename_out +'.mat')
                if proc:
                    # Initialize variables
                    MICE = []
                    data_all = None
                    data_all_reps = None
                    reds_all = None
                    mean_reliab_all = None
                    SNR_all = None
                    mouse = None
                    all_rs = None
        
                    # Find data files
                    data_dir = os.path.join(gen_path, root_dir, subdir, subsubdir)
                    data_files = [f for f in os.listdir(data_dir) if f.endswith('_Res.mat')]
                    for filename in data_files:
                        MICE.append({'filename': os.path.join(data_dir, filename)})
        
                    # Process each data file
                    for mouse_idx, mouse_data in enumerate(MICE):
                        filename = mouse_data['filename']
                        try:
                            # Load data
                            with open(filename, 'rb') as f:
                                info = mat73.loadmat(f)['info']

                            # Define training and validation stimuli based on data length
                            if len(info['StimTimes']) == 1000:
                                train_stims = np.arange(1, 651)
                                val_stims = np.arange(651, 1000, 10)
                                reps = np.arange(0, 10)
                            else:
                                train_stims = np.arange(1, 3601)
                                val_stims = np.arange(3601, 4001, 10)
                                reps = np.arange(0, 10)
                                
                            # Load and process ROI data
                            with open(filename, 'rb') as f:
                                data = mat73.loadmat(f)

                            if 'region' in data['ABAroi']:
                                good_roi = np.where(np.asarray(data['ABAroi']['region'] == area_idx+1))
                                sel_rois = True
                            else:
                                good_roi = 2
                                sel_rois = False

                            if np.sum(good_roi) > 1:
                                tb = data['Res']['ax']
                                gt = np.logical_and(tb > 0.01, tb < 0.5)
                                base_t = np.logical_and(tb > -0.2, tb < 0)
                                stims = info['Stim']['Log']
                                if sel_rois:
                                    resp = np.squeeze(data['Res']['CaDeconCorrected'][:, :, good_roi])
                                else:
                                    resp = np.squeeze(data['Res']['CaDeconCorrected'])

                                # Regress out speed
                                #resp_temp = np.nan * np.ones(resp.shape)
                                #for i in range(resp.shape[0]):
                                #    for j in range(resp.shape[2]):
                                #        _, _, r = stats.linregress(resp[i, :, j], np.vstack([np.ones(resp.shape[1]), data['Res']['speed'][i, :]]).T)
                                #        resp_temp[i, :, j] = r
    
                                # Define validation trials
                                val_trials = []
                                for i in range(len(val_stims)):
                                    val_trials.append(np.arange(val_stims[i], val_stims[i] + 10))
                                val_trials = np.asarray(val_trials)

                                # Calculate base signal and response
                                base = np.squeeze(np.nanstd(resp[base_t, :, :], axis=(0,1)))
                                signal = np.squeeze(np.nanmax(np.nanmean(resp, axis=1), axis=0))

                                # Normalize response and calculate data
                                norm_resp = []
                                for i in range(resp.shape[2]):
                                    norm_resp.append(resp[:, :, i] / signal[i])
                                norm_resp = np.moveaxis(np.asarray(norm_resp),0,-1)

                                data_v1 = []
                                for i in range(len(train_stims)):
                                    data_v1.append(np.nanmean(np.squeeze(norm_resp[gt, np.where(stims == train_stims[i]), :]), axis=0))

                                val_idx = np.arange(val_trials.min(),val_trials.max())
                                ind = 0
                                all_temps = []
                                for i in range(val_trials.shape[0]):
                                    temp = []
                                    for j in range(val_trials.shape[1]):
                                        temp.append(np.squeeze(norm_resp[gt, np.where(stims == val_trials[ind, j]), :]))
                                    
                                    temp = np.asarray(temp)
                                    all_temps.append(np.squeeze(np.nanmean(temp, axis=1)))
                                    data_v1.append(np.squeeze(np.nanmean(temp,axis=(0,1))))
                                    ind += 1
                                    
                                #print(np.asarray(all_reps).shape)
                                #print(np.asarray(all_temps).shape)
                                all_temps = np.asarray(all_temps)
                                all_reps = np.moveaxis(np.moveaxis(np.asarray(all_temps),0,-1),1,0)
                                data_v1 = np.transpose(np.asarray(data_v1))
                                trialvals = np.arange(1, data_v1.shape[1] + 1)
    
                                # Calculate reliability
                                reliab = []
                                for i in range(all_temps.shape[-1]):
                                    temp = []
                                    for j in range(all_temps.shape[1]):
                                        temp.append(np.corrcoef(np.nanmean(np.squeeze(all_temps[:, reps != j, i]), axis=1),np.squeeze(all_temps[:, j, i]))[0, 1])
                                    reliab.append(np.asarray(temp))
                                reliab = np.asarray(reliab)
                                mean_reliab = np.nanmean(reliab, axis=1)
         
                                # Calculate SNR (Signal to Noise Ratio)
                                SNR = signal / base
                                
                                # Look for red rois
                                if hasattr(info['rois'],'red'):
                                    reds = np.array([roi['red'] for roi in info['rois']])
                                else:
                                    reds = np.zeros(resp.shape[2])
                                temp_m = np.squeeze(np.ones((resp.shape[2], 1)) * (mouse_idx + 1))

                                # Compile data
                                if data_all is None:
                                    data_all = data_v1
                                else:
                                    data_all = np.vstack((data_all, data_v1))
                                
                                if data_all_reps is None:
                                    data_all_reps = all_reps
                                else:
                                    data_all_reps = np.vstack((data_all_reps, all_reps))
                                
                                if mean_reliab_all is None:
                                    mean_reliab_all = mean_reliab
                                else:
                                    mean_reliab_all = np.concatenate((mean_reliab_all, mean_reliab))

                                if reds_all is None:
                                    reds_all = reds
                                else:
                                    reds_all = np.concatenate((reds_all, reds))
                                
                                if SNR_all is None:
                                    SNR_all = SNR
                                else:
                                    SNR_all = np.concatenate((SNR_all, SNR))

                                if mouse is None:
                                    mouse = temp_m
                                else:
                                    mouse = np.concatenate((mouse, temp_m))

                                #all_rs = np.vstack([all_rs, resp_temp])
        
                        except Exception as e:
                            print(f"Error processing file {filename}: {e}")
        
                    if mean_reliab_all is not None:
                        # Select reliable neurons and save data
                        group_names.append(prefix + '_' +  subdir[:-1] + '_' + area)
                        dates.append(date)
                        cond_names.append(subsubdir[:-1])
                        good_neurons = mean_reliab_all > 0.1  # Adjust threshold for good neurons here
                        data_v1 = data_all[good_neurons, :]
                        mean_reliab = mean_reliab_all[good_neurons]
                        SNR = SNR_all[good_neurons]
                        reds = reds_all[good_neurons]
                        mouse = mouse[good_neurons]
                        #all_rs = all_rs[good_neurons, :]
    
                        output  =  {'val_idx': val_idx, 'trialvals': trialvals, 'mouse': mouse,
                                             'good_neurons': good_neurons, 'data_v1': data_v1,
                                             'val_stims': val_stims, 'mean_reliab': mean_reliab,
                                             'SNR': SNR, 'data_all_reps': data_all_reps,
                                             'reds': reds, 'MICE': MICE} #all_rs': all_rs,
                        savemat(filename_all, output) 
                else:
                    if os.path.isfile(filename_all):
                        # Save names for later
                        group_names.append(prefix + '_' +  subdir[:-1] + '_' + area)
                        dates.append(date)
                        cond_names.append(subsubdir[:-1])

    return group_names, dates, cond_names


def extract_data(gen_path,group_name,date_,condition=None):

    if condition is None:
        condition = ''
    else:
        condition = '_' + condition
    
    filename = os.path.join(gen_path,'data_NPC_'+ group_name + condition + '.pkl')
    data_path = os.path.join(gen_path,group_name + condition +'_'+ date_ +'.mat')
    # load data
    #data_dict = {}
    #f = h5py.File(data_path,'r')
    #for k, v in f.items():
    #    data_dict[k] = np.array(v)
    print(data_path)

    data_dict = loadmat(data_path)
    idxs_data = data_dict['trialvals'].astype(int)
    idxs_val = data_dict['val_idx'].astype(int)
    idxs_pics = data_dict['val_stims'][0].astype(int)
    data = np.moveaxis(data_dict['data_v1'],0,-1)
    good_neurons = data_dict['good_neurons']
    mean_reliab = data_dict['mean_reliab']
    #reds = data_dict['reds']
    reds = []
    snr = data_dict['SNR']
    
    
    if len(idxs_pics) == 35:
        ### KOEN:
        tf_path = r'D:\Paolo\mice_pics\\'
        imgs_path = r'D:\Paolo\mice_pics\pics\\'
    else:
        ### HUUB:
        tf_path = r'D:\Paolo\mice_pics_Huub_squared\\'
        imgs_path = r'D:\Paolo\mice_pics_Huub_squared\pics\\'
    
    n_neurons = data.shape[1]
    val_stims = np.where(np.isin(idxs_data,idxs_val).squeeze())
    train_stims = np.where(~np.isin(idxs_data,idxs_val).squeeze())

    val_data = data[val_stims]
    train_data = data[train_stims]
    
    def crop_reflection(image):
        return crop(image, 0, 0, 1080, 1080)
    
    transform = transforms.Compose([#transforms.Lambda(crop_reflection),
                                    transforms.Resize(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])])
    dataset = datasets.ImageFolder(tf_path, transform=transform)
    for root, _, fnames in sorted(os.walk(imgs_path, followlinks=True)):
        sort_fnames = sorted(fnames)
        
    #train data
    pics_idx = []
    for n in range(train_data.shape[0]):
        pics_idx.append(n)
    temp_subset = torch.utils.data.Subset(dataset, pics_idx)
    dataloader = torch.utils.data.DataLoader(temp_subset, batch_size=train_data.shape[0], shuffle=False)
    img_data, junk = next(iter(dataloader))

    #val data
    val_pics_idx = []
    for n in range(val_data.shape[0]):
        val_pics_idx.append(idxs_pics[n]-1)
    val_temp_subset = torch.utils.data.Subset(dataset, val_pics_idx)
    val_dataloader = torch.utils.data.DataLoader(val_temp_subset, batch_size=val_data.shape[0], shuffle=False)
    val_img_data, junk = next(iter(val_dataloader))
    
    output = {'img_data':img_data, 'val_img_data':val_img_data,
              'train_data':train_data, 'val_data':val_data, 
              'good_neurons':good_neurons,'mean_reliab':mean_reliab,
              'reds':reds, 'snr':snr}
     
    f = open(filename,"wb")
    pickle.dump(output,f)
    f.close()


def train_neural_model(gen_path,group_name,condition,layer,layers=[]):
    
    if layer == 'False':
        grid_search = True
        train_single_layer = False
        layers = layers[1:]
    elif layer == 'True':
        train_single_layer = True
        grid_search = True
        layers = layers[1:]
    else:
        grid_search = False
        
    restart = False
        
    # hyperparameters
    seed = 0
    nb_epochs = 600
    save_epochs = 100
    grid_epochs = 200
    grid_save_epochs = 200
    batch_size = 100
    lr_decay_gamma = 1/3
    lr_decay_step = 3*save_epochs
    backbone = 'inception_v1'
    
    learning_rates = [1e-2,1e-3,1e-4]
    smooth_weights = [0.01,0.05,0.1]
    sparse_weights = [1e-4]
    weight_decays = [0.001,0.005,0.01,0.05]
    
    # paths + filenames
    grid_filename = os.path.join(gen_path,'snapshots\grid_search_'+ group_name + '_' + condition +'.pkl')
    data_filename = os.path.join(gen_path,'data_NPC_'+ group_name + '_' + condition + '.pkl')
    current_datetime = time.strftime("%Y-%m-%d_%H_%M_%S", time.gmtime())
    Path(gen_path + '\\training_data\\').mkdir(parents=True, exist_ok=True)
    loss_plot_path = os.path.join(gen_path,'training_data\\training_loss_classifier_' + current_datetime + '.png')
    
    # load_snapshot = True
    load_snapshot = False
    GPU = torch.cuda.is_available()
    if GPU:
        torch.cuda.set_device(0)
    Path(gen_path + '\\snapshots\\').mkdir(parents=True, exist_ok=True)
    snapshot_pattern =  os.path.join(gen_path, 'neural_model_' + backbone + '_' + layer + '.pt')
    
    # load data
    f = open(data_filename,"rb")
    cc = pickle.load(f)
    
    train_data = cc['train_data']
    val_data = cc['val_data']
    img_data = cc['img_data']
    val_img_data = cc['val_img_data']
    n_neurons = train_data.shape[1]

    if grid_search:
        if os.path.isfile(grid_filename):
            grid_search = False
            restart = True
    
    if n_neurons>1:
        ######
        # Grid search:
        ######
        iter1_done = False
        if grid_search:
            params = []
            val_corrs = []
            for layer in layers:
                print('======================')
                print('Backbone: ' + layer)
                for learning_rate in learning_rates:
                    for smooth_weight in smooth_weights:
                        for sparse_weight in sparse_weights:
                            for weight_decay in weight_decays:
                                
                                set_seed(seed)
                                if iter1_done:
                                    del pretrained_model
                                    del net
                                    del criterion
                                    del optimizer
                                    del scheduler
                                    
                                iter1_done = True
        
                                # model, wrapped in DataParallel and moved to GPU
                                pretrained_model = inceptionv1(pretrained=True)
                                if GPU:
                                    net = Model(pretrained_model,layer,n_neurons,torch.device("cuda:0" if GPU else "cpu"))
                                    net.initialize()
                                    net = torch.nn.DataParallel(net)
                                    net = net.cuda()
                                else:
                                    net = Model(pretrained_model,layer,n_neurons,torch.device("cuda:0" if GPU else "cpu"))
                                    net.initialize()
                                    print('Initialized using Xavier')
                                    net = torch.nn.DataParallel(net)
                                    print("Training on CPU")
        
                                # loss function
                                criterion = torch.nn.MSELoss()
        
                                # optimizer and lr scheduler
                                optimizer = torch.optim.Adam(
                                    [net.module.w_s,net.module.w_f],
                                    lr=learning_rate,
                                    weight_decay=weight_decay)
                                scheduler = torch.optim.lr_scheduler.StepLR(
                                    optimizer,
                                    step_size=lr_decay_step,
                                    gamma=lr_decay_gamma)
        
                                cum_time = 0.0
                                cum_time_data = 0.0
                                cum_loss = 0.0
                                optimizer.step()
                                
                                # test
                                #torch.cuda.empty_cache()
                                
                                for epoch in range(grid_epochs):
                                    # adjust learning rate
                                    scheduler.step()
                                    torch.cuda.empty_cache()
                                    # get the inputs & wrap them in tensor
                                    batch_idx = np.random.choice(np.linspace(0,train_data.shape[0]-1,train_data.shape[0]), 
                                                                 size=batch_size,replace=False).astype('int')
        
                                    if GPU:
                                        neural_batch = torch.tensor(train_data[batch_idx,:]).cuda()
                                        img_batch = img_data[batch_idx,:].cuda()
                                    else:
                                        neural_batch = torch.tensor(train_data[batch_idx,:])
                                        img_batch = img_data[batch_idx,:]
        
                                    # forward + backward + optimize
                                    tic = time.time()
                                    optimizer.zero_grad()
                                    outputs = net(img_batch).squeeze()
                                                                        
                                    loss = criterion(outputs, neural_batch.float()) + smoothing_laplacian_loss(net.module.w_s, 
                                                                                            torch.device("cuda:0" if GPU else "cpu"), 
                                                                                            weight=smooth_weight) \
                                                                                    + sparse_weight * torch.norm(net.module.w_f,1)
        
                                    loss.backward()
                                    optimizer.step()
                                    toc = time.time()
        
                                    cum_time += toc - tic
                                    cum_loss += loss.data.cpu()
        
                                    # output & test
                                    if epoch % grid_save_epochs == grid_save_epochs - 1:
                                        torch.cuda.empty_cache()
                                        batch_idx = np.random.choice(np.linspace(0,train_data.shape[0]-1,train_data.shape[0]), 
                                                                 size=batch_size,replace=False).astype('int')
                                        if GPU:
                                            neural_batch = torch.tensor(train_data[batch_idx,:]).cuda()
                                            img_batch = img_data[batch_idx,:].cuda()
                                        else:
                                            neural_batch = torch.tensor(train_data[batch_idx,:])
                                            img_batch = img_data[batch_idx,:]
                                        
                                        val_outputs = net(img_batch).squeeze()
                                        corrs = []
                                        for n in range(val_outputs.shape[1]):
                                            corrs.append(np.corrcoef(val_outputs[:,n].cpu().detach().numpy(),neural_batch[:,n].squeeze().cpu().detach().numpy())[1,0])
                                        val_corr = np.median(corrs)
        
                                        # print and plot time / loss
                                        print('======')
                                        print(time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()))
                                        no_epoch = epoch / (grid_save_epochs - 1)
                                        mean_loss = cum_loss / float(grid_save_epochs)
                                        if mean_loss.is_cuda:
                                            mean_loss = mean_loss.data.cpu()
                                        cum_time = 0.0
                                        cum_loss = 0.0
        
        
                                params.append([layer,learning_rate,smooth_weight,sparse_weight,weight_decay])
                                val_corrs.append(val_corr)
                                #print('======================')
                                print(f'learning rate: {learning_rate}')
                                print(f'smooth weight: {smooth_weight}')
                                print(f'sparse weight: {sparse_weight}')
                                print(f'weight decay: {weight_decay}')
                                print(f'Validation corr: {val_corr:.3f}')
                                #print('======')
        
            # save params
            output = {'val_corrs':val_corrs, 'params':params}
            f = open(grid_filename,"wb")
            pickle.dump(output,f)
            f.close()
            
            # extract winning params
            val_corrs = np.array(val_corrs)
            layer = params[np.where(val_corrs==np.nanmax(val_corrs))[0][0].astype('int')][0]
            learning_rate = params[np.where(val_corrs==np.nanmax(val_corrs))[0][0].astype('int')][1]
            smooth_weight = params[np.where(val_corrs==np.nanmax(val_corrs))[0][0].astype('int')][2]
            sparse_weight = params[np.where(val_corrs==np.nanmax(val_corrs))[0][0].astype('int')][3]
            weight_decay = params[np.where(val_corrs==np.nanmax(val_corrs))[0][0].astype('int')][4]
        
            # print winning params
            print('======================')
            print('Best backbone is: ' + layer)
            print('Best learning rate is: ' + str(learning_rate))
            print('Best smooth weight is: ' + str(smooth_weight))
            print('Best sparse weight is: ' + str(sparse_weight))
            print('Best weight decay is: ' + str(weight_decay))
            
            # save params
            output = {'val_corrs':val_corrs, 'params':params}
            f = open(grid_filename,"wb")
            pickle.dump(output,f)
            f.close()
        else:   
            print('======================')
            print('Backbone: ' + layer)
            f = open(grid_filename,"rb")
            cc = pickle.load(f)
            val_corrs = cc['val_corrs']
            params = cc['params']
            if restart:
                layer = params[np.where(val_corrs==np.nanmax(val_corrs))[0][0].astype('int')][0]
            all_layers =  np.asarray(params)[:,0]
            val_corrs = np.array(val_corrs)
            max_layer = (np.nanmax(val_corrs[all_layers == layer]))
            good_params = params[np.where(val_corrs==max_layer)[0][0].astype('int')]
            learning_rate = good_params[1]
            smooth_weight = good_params[2]
            sparse_weight = good_params[3]
            weight_decay = good_params[4]
            train_single_layer = True
        
        ######
        # Final training!!
        ######
        
        
        snapshot_path = os.path.join(gen_path, group_name + '_' + condition + '_' + layer + '_neural_model.pt')
        layer_filename = os.path.join(gen_path, 'val_corr_'+ group_name + '_' + condition + '_' + layer + '.pkl')

        if os.path.isfile(layer_filename):
            train_single_layer = False

        if train_single_layer:
            
            # model, wrapped in DataParallel and moved to GPU
            set_seed(seed)
            pretrained_model = inceptionv1(pretrained=True)
            if GPU:
                net = Model(pretrained_model,layer,n_neurons,torch.device("cuda:0" if GPU else "cpu"))
                if load_snapshot:
                    net.load_state_dict(torch.load(
                        snapshot_path,
                        map_location=lambda storage, loc: storage
                    ))
                    print('Loaded snap ' + snapshot_path)
                else:
                    net.initialize()
                    print('Initialized using Xavier')
                net = torch.nn.DataParallel(net)
                net = net.cuda()
                print("Training on {} GPU's".format(torch.cuda.device_count()))
            else:
                net = Model(pretrained_model,layer,n_neurons,torch.device("cuda:0" if GPU else "cpu"))
                if load_snapshot:
                    net.load_state_dict(torch.load(
                        snapshot_path,
                        map_location=lambda storage, loc: storage
                    ))
                    print('Loaded snap ' + snapshot_path)
                else:
                    net.initialize()
                    print('Initialized using Xavier')
                net = torch.nn.DataParallel(net)
                print("Training on CPU")
        
            # loss function
            criterion = torch.nn.MSELoss()
        
            # optimizer and lr scheduler
            optimizer = torch.optim.Adam(
                [net.module.w_s,net.module.w_f],
                lr=learning_rate,
                weight_decay=weight_decay)
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=lr_decay_step,
                gamma=lr_decay_gamma)
        
            # figure for loss function
            fig = plt.figure()
            axis = fig.add_subplot(111)
            axis.set_xlabel('epoch')
            axis.set_ylabel('loss')
            axis.set_yscale('log')
            plt_line, = axis.plot([], [])
        
            cum_time = 0.0
            cum_time_data = 0.0
            cum_loss = 0.0
            optimizer.step()
            for epoch in range(nb_epochs):
                # adjust learning rate
                scheduler.step()
        
                # get the inputs & wrap them in tensor
                batch_idx = np.random.choice(np.linspace(0,train_data.shape[0]-1,train_data.shape[0]), 
                                             size=batch_size,replace=False).astype('int')
                if GPU:
                    neural_batch = torch.tensor(train_data[batch_idx,:]).cuda()
                    val_neural_data = torch.tensor(val_data).cuda()
                    img_batch = img_data[batch_idx,:].cuda()
                else:
                    neural_batch = torch.tensor(train_data[batch_idx,:])
                    val_neural_data = torch.tensor(val_data)
                    img_batch = img_data[batch_idx,:]
        
                # forward + backward + optimize
                tic = time.time()
                optimizer.zero_grad()
                outputs = net(img_batch).squeeze()
                loss = criterion(outputs, neural_batch.float()) + smoothing_laplacian_loss(net.module.w_s, 
                                                                                  torch.device("cuda:0" if GPU else "cpu"), 
                                                                                  weight=smooth_weight) \
                                                                + sparse_weight * torch.norm(net.module.w_f,1)
        
                loss.backward()
                optimizer.step()
                toc = time.time()
        
                cum_time += toc - tic
                cum_loss += loss.data.cpu()
        
                # output & test
                if epoch % save_epochs == save_epochs - 1:
        
                    val_outputs = net(val_img_data).squeeze()
                    val_loss = criterion(val_outputs, val_neural_data)
                    corrs = []
                    for n in range(val_outputs.shape[1]):
                        corrs.append(np.corrcoef(val_outputs[:,n].cpu().detach().numpy(),val_data[:,n])[1,0])
                    val_corr = np.median(corrs)
        
                    tic_test = time.time()
                    # print and plot time / loss
                    print('======================')
                    print(time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()))
                    no_epoch = epoch / (save_epochs - 1)
                    mean_time = cum_time / float(save_epochs)
                    mean_loss = cum_loss / float(save_epochs)
                    if mean_loss.is_cuda:
                        mean_loss = mean_loss.data.cpu()
                    cum_time = 0.0
                    cum_loss = 0.0
                    print(f'epoch {int(epoch)}/{nb_epochs} mean time: {mean_time:.3f}s')
                    print(f'epoch {int(epoch)}/{nb_epochs} mean loss: {mean_loss:.3f}')
                    print(f'epoch {int(no_epoch)} validation loss: {val_loss:.3f}')
                    print(f'epoch {int(no_epoch)} validation corr: {val_corr:.3f}')
                    plt_line.set_xdata(np.append(plt_line.get_xdata(), no_epoch))
                    plt_line.set_ydata(np.append(plt_line.get_ydata(), mean_loss))
                    axis.relim()
                    axis.autoscale_view()
        
                    fig.savefig(loss_plot_path)
        
                    print('======================')
                    print('Test time: ', time.time()-tic_test)
        
            # save final val corr
            output = {'val_corrs':corrs}
            f = open(layer_filename,"wb")
            pickle.dump(output,f)
            f.close()
        
            # save the weights, we're done!
            os.makedirs(os.path.dirname(snapshot_path), exist_ok=True)
            torch.save(net.module.state_dict(), snapshot_path)


### 2nd level analyses + MEIs:

def load_mouse_model(gen_path,group_name,condition):
    data_filename = os.path.join(gen_path, 'snapshots/grid_search_'+ group_name + condition + '.pkl')
    f = open(data_filename,"rb")
    cc = pickle.load(f)
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

def plot_layer_corrs(gen_path,group_name,layers,condition):
    all_corrs = []
    for layer in layers:
        layer_filename = os.path.join(gen_path, 'val_corr_'+ group_name + condition + '_' + layer + '.pkl')
        f = open(layer_filename,"rb")
        cc = pickle.load(f)
        val_corrs = cc['val_corrs']
        all_corrs.append(val_corrs)
        
    all_corrs = np.array(all_corrs)
    fig1, ax = plt.subplots()
    c = '#3399ff'
    ax.set_title('Layers:')
    ax.boxplot(all_corrs.transpose(), labels=layers, notch=True,
               widths=.5,showcaps=False, whis=[2.5,97.5],patch_artist=True,
               showfliers=False,boxprops=dict(facecolor=c, color=c),
               capprops=dict(color=c),whiskerprops=dict(color=c),
               flierprops=dict(color=c, markeredgecolor=c),medianprops=dict(color='w',linewidth=2))
    ax.set_ylabel('Cross-validated Pearson r')
    ax.set_ylim([0,1])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.show()

    return all_corrs

def compute_val_corrs(gen_path,group_name,mouse_model,condition):
    data_filename = os.path.join(gen_path, 'data_NPC_'+ group_name + condition +'.pkl')
    f = open(data_filename,"rb")
    cc = pickle.load(f)

    val_img_data = cc['val_img_data']
    val_outputs = mouse_model(val_img_data).squeeze()
    val_data = cc['val_data']
    corrs = []
    for n in range(val_outputs.shape[1]):
        corrs.append(np.corrcoef(val_outputs[:,n].cpu().detach().numpy(),val_data[:,n])[1,0])
    return corrs

def good_neurons(mouse_model,n_neurons,corrs,make_plots=True):
    z = 1
    all_good_rfs = []
    goods = []
    for n in range(n_neurons):
        mouse_model_rf = np.abs(np.reshape(mouse_model.w_s[n].squeeze().detach().cpu().numpy(),
                          [np.sqrt(mouse_model.w_s.shape[2]).astype('int'),np.sqrt(mouse_model.w_s.shape[2]).astype('int')]))
        if corrs[n] > 0:
            z += 1
            goods.append(n)
            mouse_model_rf_norm = (mouse_model_rf+np.nanmin(np.abs(mouse_model_rf)))/(np.nanmax(np.abs(mouse_model_rf))+np.nanmin(np.abs(mouse_model_rf)))
            all_good_rfs.append(mouse_model_rf_norm)
            if make_plots:
                print(z,corrs[n])
                plt.imshow(mouse_model_rf, cmap='seismic')
                plt.colorbar()
                plt.show()
                plt.plot(mouse_model.w_f[0,n].squeeze().detach().cpu().numpy())
                plt.show()

    goods = np.array(goods)
    all_good_rfs = np.array(all_good_rfs)
    return goods,all_good_rfs

def gaussian_RFs(gen_path,all_good_rfs,goods,corrs,all_corrs,pixperdeg,stim_size,mask_size,mouse_model,group_name,condition):
    centrex = []
    centrey = []
    szx = []
    szy = []
    szdeg = []
    masks = []
    sparsity = []
    all_feats = []
    pixperdeg_reduced = all_good_rfs[0].shape[0]/all_good_rfs*pixperdeg
    scaling_f = stim_size/all_good_rfs[0].shape[0]
    scaling_mask = mask_size/all_good_rfs[0].shape[0]
    pixperdeg_reduced_mask = all_good_rfs[0].shape[0]/mask_size*pixperdeg
    ax = plt.gca()
    ax.set_title('Spatial RFs:')
    for n in range(len(goods)):

        # fit 2d Gaussian to W_s
        data = all_good_rfs[n]
        params = fitgaussian(data)
        fit = gaussian(*params)
        plt.imshow(data, cmap=plt.cm.gist_earth_r)
        (height, y, x, width_y, width_x) = params # x and y are shifted in img coords
        circle = Ellipse((x, y), 2*1.65*width_x, 2*1.65*width_y, edgecolor='r', facecolor='None', clip_on=True)
        ax.add_patch(circle)
        centrex.append(x*scaling_f)
        centrey.append(y*scaling_f)
        szx.append(2*1.65*width_x*scaling_f)
        szy.append(2*1.65*width_y*scaling_f)
        szdeg.append(np.nanmean((2*1.65*width_x/pixperdeg_reduced,2*1.65*width_y/pixperdeg_reduced)))

        # create binary mask summing up the estimated 2d Gaussians (95% CI)
        mask = np.zeros(shape=(mask_size,mask_size), dtype="bool")
        circ = shapely.geometry.Point((x*scaling_mask,y*scaling_mask)).buffer(1)
        ell  = shapely.affinity.scale(circ, 1.65*width_x*scaling_mask+pixperdeg_reduced_mask, 1.65*width_y*scaling_mask+pixperdeg_reduced_mask)
        ell_coords = np.array(list(ell.exterior.coords))
        cc, rr = polygon(ell_coords[:,0], ell_coords[:,1], mask.shape)
        mask[rr,cc] = True
        masks.append(mask)
        
        feats = mouse_model.w_f[0,goods[n]].squeeze().detach().cpu().numpy()
        chn_spars = (1 - (((np.sum(np.abs(feats)) / len(feats)) ** 2) / (np.sum(np.abs(feats) ** 2) / len(feats)))) / (1 - (1 / len(feats)))
        sparsity.append(chn_spars)
        all_feats.append(feats)

    plt.show()
    masks = np.array(masks)
    centrex = np.array(centrex)
    centrey = np.array(centrey)
    szx = np.array(szx)
    szy = np.array(szy)
    szdeg = np.array(szdeg)
    sparsity = np.array(sparsity)
    all_feats = np.array(all_feats)

     # plot:
    labels = ['Sparsity index','Frequency (neurons)','']
    c = '#3399ff'
    xline = [np.nanmedian(sparsity)]
    fig1, ax = plt.subplots()
    ax.hist(sparsity,bins=10,color=c)
    if len(xline) != 0:
        plt.axvline(xline[0], color='k', linestyle='dashed')
    if len(labels) != 0:
        ax.set_xlabel(labels[0])
        ax.set_ylabel(labels[1])
        ax.set_title(labels[2])
    ax.set_xlim([0,1])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_title('Feature sparsness:')
    plt.show()
    
    all_corrs = np.moveaxis(all_corrs,0,-1)
    
    output = {'goods':goods,
             'all_good_rfs':all_good_rfs,
             'cross_val_corr':corrs,
             'layers_cross_val_corr':all_corrs,
             'centrex':centrex,
             'centrey':centrey,
             'szx':szx,
             'szy':szy,
             'szdeg':szdeg,
             'all_feats':all_feats,
             'sparsity':sparsity}
    savemat(gen_path + '\\' + group_name + condition + '_good_rfs' + '.mat', output)
    return masks

def generate_MEIS(gen_path,mouse_model,group_name,goods=None,condition='',sign=1):
    if sign == 1:
        sign_label = 'MEIs/'
    elif sign == -1:
        sign_label = 'MIIs/'
    else:
        raise Exception('The variable sign must be either 1 (for MEIs) or -1 (for MIIs)')
        
    Path(gen_path + '\\' + sign_label + group_name + condition + '_fft_decorr_lr_1e-2_l2_1e-3/').mkdir(parents=True, exist_ok=True)

    if goods is None:
        goods = np.linspace(0, mouse_model.n_neurons-1, mouse_model.n_neurons).astype(int)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    mouse_model.to(device).eval()
    all_transforms = [
        transform.pad(2),
        transform.jitter(4),
        transform.random_scale([n/1000. for n in range(975, 1050)]),
        transform.random_rotate(list(range(-5,5))),
        transforms.Grayscale(3),
    ]
    batch_param_f = lambda: param.image(128, batch=1, fft=True, decorrelate=True)
    cppn_opt = lambda params: torch.optim.Adam(params, 1e-2, weight_decay=1e-3)

    for chn in range(len(goods)):
        obj = objectives.channel("output", int(goods[chn]))*sign
        _ = render.render_vis(mouse_model, obj, batch_param_f, cppn_opt, transforms=all_transforms, 
                              show_inline=True, thresholds=(50,))
        for n in range(_[0].shape[0]):
            filename = os.path.join(gen_path, sign_label + group_name + condition + '_fft_decorr_lr_1e-2_l2_1e-3/' + str(goods[chn]) + '_' + str(n) + '.bmp')
            temp = _[0][n]#*np.repeat(np.expand_dims(masks[chn],-1),3,axis=-1)+np.repeat((1-np.expand_dims(masks[chn],-1))*.5,3,axis=-1)
            temp_pic = Image.fromarray(np.uint8((temp)*255))
            temp_pic = temp_pic.resize((500,500),Image.BILINEAR)
            temp_pic.save(filename)

def generate_single_MEIS(gen_path,all_corrs,group_name,n_neurons,layers,goods=None,condition='',sign=1):
    if sign == 1:
        sign_label = 'MEIs_single/'
    elif sign == -1:
        sign_label = 'MIIs_single/'
    else:
        raise Exception('The variable sign must be either 1 (for MEIs) or -1 (for MIIs)')
        
    Path(gen_path + '\\' + sign_label + group_name + condition + '_fft_decorr_lr_1e-2_l2_1e-3/').mkdir(parents=True, exist_ok=True)

    if goods is None:
        goods = np.linspace(0, all_corrs.shape[1]-1, all_corrs.shape[1]).astype(int)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    layers_max = (all_corrs[:,goods] == np.max(all_corrs[:,goods],axis=0))
    pretrained_model = inceptionv1(pretrained=True)
    for lay in range(all_corrs.shape[0]):
        goods_lay = goods[np.where(layers_max[lay,:])]
        layer = layers[lay]
        if len(goods_lay) > 0:
            mouse_model = Model(pretrained_model,layer,n_neurons,device='cpu')
            mouse_model.load_state_dict(torch.load(gen_path + '\\' + group_name + condition + '_' + layer + '_neural_model.pt',map_location=torch.device('cpu')))
            mouse_model.to(device).eval()
            
            all_transforms = [
                transform.pad(2),
                transform.jitter(4),
                transform.random_scale([n/1000. for n in range(975, 1050)]),
                transform.random_rotate(list(range(-5,5))),
                transforms.Grayscale(3),
            ]
            batch_param_f = lambda: param.image(128, batch=1, fft=True, decorrelate=True)
            cppn_opt = lambda params: torch.optim.Adam(params, 1e-2, weight_decay=1e-3)
        
            for chn in range(len(goods_lay)):
                obj = objectives.channel("output", int(goods_lay[chn]))*sign
                _ = render.render_vis(mouse_model, obj, batch_param_f, cppn_opt, transforms=all_transforms, 
                                      show_inline=True, thresholds=(50,))
                for n in range(_[0].shape[0]):
                    filename = os.path.join(gen_path, sign_label + group_name + condition + '_fft_decorr_lr_1e-2_l2_1e-3/' + str(goods_lay[chn]) + '_' + str(n) + '.bmp')
                    temp = _[0][n]#*np.repeat(np.expand_dims(masks[chn],-1),3,axis=-1)+np.repeat((1-np.expand_dims(masks[chn],-1))*.5,3,axis=-1)
                    temp_pic = Image.fromarray(np.uint8((temp)*255))
                    temp_pic = temp_pic.resize((500,500),Image.BILINEAR)
                    temp_pic.save(filename)
            mouse_model.to('cpu').eval()


def generate_surround_MEIS(gen_path,mouse_model,group_name,goods=None,condition=''):
    Path(gen_path + '/surrMEIs/' + group_name + condition + '_fft_decorr_lr_1e-2_l2_1e-3/').mkdir(parents=True, exist_ok=True)

    if goods is None:
        goods = np.linspace(0, mouse_model.n_neurons-1, mouse_model.n_neurons).astype(int)
        
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    mouse_model.to(device).eval()
    all_transforms = [
        transform.pad(2),
        transform.jitter(4),
        transform.random_scale([n/1000. for n in range(975, 1050)]),
        transform.random_rotate(list(range(-5,5))),
        transforms.Grayscale(3),
    ]
    batch_param_f = lambda: param.image(128, batch=1, fft=True, decorrelate=True)
    cppn_opt = lambda params: torch.optim.Adam(params, 1e-2, weight_decay=1e-3)

    for chn in range(len(goods)):
        obj = objectives.channel("output", int(goods[chn]))
        mei = render.render_vis(mouse_model, obj, batch_param_f, cppn_opt, transforms=all_transforms, 
                              show_inline=True, thresholds=(50,))

        mask = mouse_model.w_s[chn].squeeze()
        mask = torch.reshape(mask,(np.sqrt(mask.shape[0]).astype('int'),np.sqrt(mask.shape[0]).astype('int')))
        mask = np.abs(nnf.interpolate(mask[None,None,:],mei[0].shape[1]).squeeze().cpu().detach().numpy())
        mask = smoothing(mask>(np.nanstd(mask)*3),1)
        mask = np.repeat(mask[:,:,np.newaxis],3,2)
   
        all_transforms = [
            occlude_pic(mei[0], mask,device),
            transform.pad(2),
            transform.jitter(4),
            transform.random_scale([n/1000. for n in range(975, 1050)]),
            transform.random_rotate(list(range(-5,5))),
            transforms.Grayscale(3),
        ]
    
        for i in range(2):
            if i:
                obj = objectives.channel("output", int(goods[chn]))
                filename = os.path.join(gen_path, 'surrMEIs/' + group_name + condition + '_fft_decorr_lr_1e-2_l2_1e-3/' + str(goods[chn]) + '_positive_Surround.bmp')
            else:
                obj = -objectives.channel("output", int(goods[chn]))
                filename = os.path.join(gen_path, 'surrMEIs/' + group_name + condition + '_fft_decorr_lr_1e-2_l2_1e-3/' + str(goods[chn]) + '_negative_Surround.bmp')


            _ = render.render_vis(mouse_model, obj, batch_param_f, cppn_opt, transforms=all_transforms, 
                              show_inline=True, thresholds=(50,))
            temp = _[0][0]
            temp_pic = Image.fromarray(np.uint8((temp)*255))
            temp_pic = temp_pic.resize((500,500),Image.BILINEAR)
            temp_pic.save(filename)
                       
def ori_tuning(gen_path,stim_size,wavelengths,orientations,phases,contrasts,mouse_model,group_name,goods,condition):
    # load data:
    data_path = os.path.join(gen_path, group_name + condition + '_good_rfs' + '.mat')
    data_dict = {}
    f = loadmat(data_path)
    for k, v in f.items():
        data_dict[k] = np.array(v)
    centrex = data_dict['centrex'].astype(int)[0]
    centrey = data_dict['centrey'].astype(int)[0]

    # define transforms:
    #transform = transforms.Compose([transforms.ToPILImage(),
    #                            transforms.Resize([224,224]),
    #                            transforms.ToTensor(),
    #                            transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                            std=[0.229, 0.224, 0.225])])
    
    transform = transforms.Compose([transforms.ToPILImage(),
                                transforms.Resize([224,224]),
                                transforms.ToTensor()])
    # RF center:
    px_x = np.nanmedian(centrex)
    px_y = np.nanmedian(centrey)

    # iterate gabors:
    stimuli = []
    all_params = []
    for w in range(len(wavelengths)):
        for o in range(len(orientations)):
            for p in range(len(phases)):
                for c in range(len(contrasts)):
                    
                    min_b = 0.5 - contrasts[c]/2
                    max_b = 0.5 + contrasts[c]/2

                    # create stimulus:
                    stimulus = gabor_patch([stim_size,stim_size], pos_yx = [px_y,px_x], 
                                           radius = stim_size, wavelength = wavelengths[w], 
                                           orientation = orientations[o], phase = phases[p],
                                           min_brightness = min_b, max_brightness = max_b)
                    stimuli.append(stimulus)
                    all_params.append([wavelengths[w],orientations[o],phases[p],contrasts[c]])

    all_params = np.array(all_params)

    # get response:
    stimuli = np.array(stimuli).squeeze()
    all_resps_tot = []
    for s in range(stimuli.shape[0]):
        temp = transform(np.moveaxis(stimuli[s]*255,0,-1).squeeze().astype(np.uint8))#.to('cpu')
        all_resps_tot.append(mouse_model(temp[None,:,:,:]).squeeze()[goods].detach().numpy())

    all_resps_tot = np.array(all_resps_tot)

    OI_sel = []
    BEST_params = []
    TUN_curves = []
    SFS_curves = []
    CNTR_curves = []
    for n in range(len(goods)):
        # reshape the overall response:
        all_resps = all_resps_tot[:,n].reshape(len(wavelengths),len(orientations),len(phases),len(contrasts))

        # compute the orientation index OI:
        norm_resp = (all_resps - np.nanmin(all_resps)) / (np.nanmax(all_resps) - np.nanmin(all_resps))
        max_idx = np.where(norm_resp == np.nanmax(norm_resp))
        if len(max_idx)>1:
            max_idx = np.array(max_idx)
            max_idx = max_idx[:,-1]

        ortho_idx = np.where((180 * orientations / math.pi) == (180 * orientations / math.pi)[max_idx[1]] + 90)
        if np.array(ortho_idx).size == 0: 
            ortho_idx = np.where((180 * orientations / math.pi) == (180 * orientations / math.pi)[max_idx[1]] - 90)
        chn_sel = (np.nanmax(norm_resp) - norm_resp[max_idx[0],ortho_idx,max_idx[2],max_idx[3]]) / (np.nanmax(norm_resp) + norm_resp[max_idx[0],ortho_idx,max_idx[2],max_idx[3]])
        OI_sel.append(np.squeeze(chn_sel))
        
        # find best parameters (= max response)
        #print(np.where(all_resps_tot == np.nanmax(all_resps))[0][0])
        par_max = all_params[int(np.where(all_resps_tot == np.nanmax(all_resps))[0][0]),:]
        BEST_params.append(par_max)
        
        # output all tuning curve
        ori_curve = np.squeeze(norm_resp[max_idx[0],:,max_idx[2],max_idx[3]])
        TUN_curves.append(ori_curve)
        
        #sf_curve = np.squeeze(norm_resp[:,max_idx[1],max_idx[2],max_idx[3]])
        sf_curve = np.squeeze(all_resps[:,max_idx[1],max_idx[2],max_idx[3]])
        SFS_curves.append(sf_curve)
        
        con_curve = np.nanmean(norm_resp,axis=(0,1,2))
        CNTR_curves.append(con_curve)

    OI_sel = np.array(OI_sel)
    BEST_params = np.array(BEST_params)
    TUN_curves = np.array(TUN_curves)
    CNTR_curves = np.array(CNTR_curves)

    # plot:
    labels = ['Orientation selectivity index','Frequency (neurons)','']
    c = '#3399ff'
    xline = [np.nanmedian(OI_sel)]
    fig1, ax = plt.subplots()
    ax.hist(OI_sel,bins=10,color=c)
    if len(xline) != 0:
        plt.axvline(xline[0], color='k', linestyle='dashed')
    if len(labels) != 0:
        ax.set_xlabel(labels[0])
        ax.set_ylabel(labels[1])
        ax.set_title(labels[2])
    ax.set_xlim([0,1])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_title('Orientation tuning:')
    plt.show()
    
    output = {'goods':goods,
              'OI_sel':OI_sel,
              'SFS_curves':SFS_curves,
              'TUN_curves':TUN_curves,
              'CNTR_curves':CNTR_curves,
              'BEST_params':BEST_params}
    savemat(gen_path + '\\' + group_name + condition + '_ori_sel' + '.mat', output)
    
def size_tuning(gen_path,stim_size,radii,mouse_model,group_name,goods,condition):
    # load data:
    data_path = os.path.join(gen_path, group_name + condition + '_good_rfs' + '.mat')
    data_dict = {}
    f = loadmat(data_path)
    for k, v in f.items():
        data_dict[k] = np.array(v)
    centrex = data_dict['centrex'].astype(int)[0]
    centrey = data_dict['centrey'].astype(int)[0]

    radii_f = radii.astype(int)

    data_path = os.path.join(gen_path, group_name + condition + '_ori_sel' + '.mat')
    data_dict = {}
    f = loadmat(data_path)
    for k, v in f.items():
        data_dict[k] = np.array(v)
    BEST_params = data_dict['BEST_params']

    # define transforms:
    transform = transforms.Compose([transforms.ToPILImage(),
                                transforms.Resize([224,224]),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])])
    
    SSI_sel = []
    BEST_size = []
    S_TUN_curves = []
    for n in tqdm(range(len(goods)), disable=False):
         # RF center:
        px_x = centrex[n]
        px_y = centrey[n]
        params = BEST_params[n,:]

        # iterate gabors:
        stimuli = []
        for r in range(len(radii_f)):
            # create stimulus:
            stimulus = gabor_patch([stim_size,stim_size], pos_yx = [px_y,px_x], 
                                   radius = radii_f[r], wavelength = params[0], 
                                   orientation = params[1], phase = params[2])
            stimuli.append(stimulus)

        # get response:
        stimuli = np.array(stimuli).squeeze()
        all_resps = []
        for s in range(stimuli.shape[0]):
            temp = transform(np.moveaxis(stimuli[s]*255,0,-1).squeeze().astype(np.uint8))#.to('cpu')
            all_resps.append(mouse_model(temp[None,:,:,:]).squeeze()[goods[n]].detach().numpy())

        all_resps = np.array(all_resps)

        # find best size (= max response)
        index = np.where(all_resps == np.nanmax(all_resps))[0]
        if len(index) > 1:
            index = index[0]
        rad_max = radii[int(index)]
        BEST_size.append(rad_max)
        

        # compute the surround suppression index SS:
        norm_resp = (all_resps - np.nanmin(all_resps)) / (np.nanmax(all_resps) - np.nanmin(all_resps))
        max_resp = norm_resp[int(index)]
        fin_resp = norm_resp[-1]
        if max_resp == fin_resp:
            chn_sel = 0
        else:
            chn_sel = (max_resp - fin_resp) / (max_resp + fin_resp) 
        SSI_sel.append(np.squeeze(chn_sel))
        S_TUN_curves.append(norm_resp)

    SSI_sel = np.array(SSI_sel)
    BEST_size = np.array(BEST_size)
    S_TUN_curves = np.array(S_TUN_curves)

    # plot:
    labels = ['Surround suppression index','Frequency (neurons)','']
    c = '#3399ff'
    xline = [np.nanmedian(SSI_sel)]
    fig1, ax = plt.subplots()
    ax.hist(SSI_sel,bins=10,color=c)
    if len(xline) != 0:
        plt.axvline(xline[0], color='k', linestyle='dashed')
    if len(labels) != 0:
        ax.set_xlabel(labels[0])
        ax.set_ylabel(labels[1])
        ax.set_title(labels[2])
    ax.set_xlim([0,1])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_title('Surround suppression:')
    plt.show()
    
    output = {'goods':goods,
              'SSI_sel':SSI_sel,
              'S_TUN_curves':S_TUN_curves,
              'BEST_size':BEST_size}
    savemat(gen_path + '\\' +  group_name + condition + '_size_sel' + '.mat', output)

def get_full_resps(gen_path,mouse_model,goods,group_name,condition,folder):
    mouse_model.to('cpu').eval()
    mouse_model.w_s = torch.nn.Parameter(torch.ones(mouse_model.w_s.shape),requires_grad=False)
    transform = transforms.Compose([transforms.Resize([224,224]),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])])
    dataset = datasets.ImageFolder(folder, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset.imgs), shuffle=False)
    img_data, junk = next(iter(dataloader))
    resps = mouse_model(img_data).squeeze().detach().numpy()[:,goods]
                
    output = {'goods':goods,
             'responses':resps}
    savemat(gen_path + group_name + condition + '_responses' + '.mat', output)
    return resps, img_data


def output(gen_path,group_name,condition,goods,good_neurons,reds,mean_reliab,snr,resps=False):
    # the output must be re-arranged in the same format as the input files
    # but we applied two selections (one for reliability and one for val-corr)
    #max_n = len(good_neurons)
    max_n = good_neurons.shape[1]
    tot = np.array(np.where(good_neurons.squeeze()==1)).squeeze()
    goods_ok = tot[goods]
    
    data_dict = {}
    
    data_path = os.path.join(gen_path, group_name + condition + '_good_rfs' + '.mat')
    f = loadmat(data_path)
    for k, v in f.items():
        if k[0] != '_':
            temp = np.array(v).squeeze()
            #idx = np.array(np.where(np.array(temp.shape) == np.array(goods.shape[0]))).squeeze().astype(int)
            ts = np.array(temp.shape)
            ts[0] = max_n
            temp_r = np.empty(ts)
            temp_r.fill(np.nan)
            if np.array(temp.shape[0]) == np.array(goods.shape[0]):
                temp_r[goods_ok] = temp
            else:
                temp_r[tot] = temp
            data_dict[k] = temp_r
    
    data_path = os.path.join(gen_path, group_name + condition + '_ori_sel' + '.mat')
    f = loadmat(data_path)
    for k, v in f.items():
        if k[0] != '_':
            temp = np.array(v).squeeze()
            #idx = np.array(np.where(np.array(temp.shape) == np.array(goods.shape[0]))).squeeze().astype(int)
            ts = np.array(temp.shape)
            ts[0] = max_n
            temp_r = np.empty(ts)
            temp_r.fill(np.nan)
            if np.array(temp.shape[0]) == np.array(goods.shape[0]):
                temp_r[goods_ok] = temp
            else:
                temp_r[tot] = temp
            data_dict[k] = temp_r
            
    data_path = os.path.join(gen_path, group_name + condition + '_size_sel' + '.mat')
    f = loadmat(data_path)
    for k, v in f.items():
        if k[0] != '_':
            temp = np.array(v).squeeze()
            #idx = np.array(np.where(np.array(temp.shape) == np.array(goods.shape[0]))).squeeze().astype(int)
            ts = np.array(temp.shape)
            ts[0] = max_n
            temp_r = np.empty(ts)
            temp_r.fill(np.nan)
            if np.array(temp.shape[0]) == np.array(goods.shape[0]):
                temp_r[goods_ok] = temp
            else:
                temp_r[tot] = temp
            data_dict[k] = temp_r
    
    data_dict['good_neurons'] = good_neurons
    data_dict['reds'] = reds
    data_dict['snr'] = snr
    data_dict['mean_reliab'] = mean_reliab
    if resps:
        data_path = os.path.join(gen_path, group_name + condition + '_responses' + '.mat')
        f = loadmat(data_path)
        for k, v in f.items():
            if k[0] != '_':
                temp = np.transpose(np.array(v).squeeze())
                ts = np.array(temp.shape)
                ts[0] = max_n
                temp_r = np.empty(ts)
                temp_r.fill(np.nan)
                if np.array(temp.shape[0]) == np.array(goods.shape[0]):
                    temp_r[goods_ok] = temp
                else:
                    temp_r[tot] = temp
                data_dict[k] = temp_r
    
    savemat(gen_path + '\\' +  group_name + condition + '_OUTPUT' + '.mat', data_dict)
    
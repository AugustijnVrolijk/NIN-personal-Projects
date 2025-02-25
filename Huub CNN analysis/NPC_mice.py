import npc
import numpy as np
import math

### the first time you need to format the data 
### once done, set this to false to save time
### N.B. the first time it can take a while!
format_data = False
### same goes for extract_data!
extract_data = False

### The code needs a simple but specific folder structure 
### each file is searched by looking at a path like this:
### ProjectName\\GroupName\\ConditionName\\DataName_Res.mat
### just re-arrange your data to match this before running this 

### data-constants:
prefix = 'Huub_NaturalImages' # this is the prefix for all momdel outputs
date = '2022' # use single date (e.g. year) - this does not really need to be a date, you can use any arbitrary name
gen_path = r'D:\2Pdata\Huub\Data\NaturalImages\MEI_output' #general path for the model outputs - leave the r in front of the path or you'll get an error
data_path = r'D:\2Pdata\Huub\Data\\' # general path for the neuronal data
root_dir = 'NaturalImages\\' # root folder containing the neuronal data 
#subdirs = ['darkReared\\'] # subfolders of the groups - just one if you have only one group
#subsubdirs = ['chronic\\'] # sub-sub-folders of the conditions/factors - just one if you have only one condition
#areaNames = ['VISp'] # Brain areas - if more than one, you need to have this specified in 'ABAroi' under 'region'
subdirs = ['normReared\\', 'darkReared\\'] # subfolders of the groups - just one if you have only one group
subsubdirs = ['baseline\\', 'chronic\\'] # sub-sub-folders of the conditions/factors - just one if you have only one condition
areaNames = ['VISpm', 'VISam', 'VISa', 'VISrl', 'VISal', 'VISl', 'VISp'] # Brain areas - if more than one, you need to have this specified in 'ABAroi' under 'region'

group_names,dates,cond_names = npc.format_data(format_data,prefix,date,gen_path,data_path,root_dir,subdirs,subsubdirs,areaNames)

### if you want to collect model responses from arbitrary 
### stimuli, set this to true and specify the folder path
extra_stims = False
if extra_stims:
    pics_folder = data_path + 'YourFolderName\\' # the folder needs to contain just a sub-folder (e.g. 'pics') which then contains the images


### layers for traning and testing:
###       + 'False' on top, will run the grid-search first and stop
###       + 'True' on top, will run the grid-search first and then train the best model
layers = ['True',
          'conv2d1',
          'conv2d2',
          'mixed3a',
          'mixed3b',
          'mixed4a',
          'mixed4b']

### constants for mei generation:
seed = 0
pixperdeg = 16
stim_size = 1080
mask_size = 128

### constants for orientation and size tuning:
wavelengths = np.floor(pixperdeg/np.asarray((0.08,0.1,0.2,0.3,0.4,0.6)))
orientations = np.arange(0, math.pi, math.pi/8)
phases = np.array([0, 2*math.pi])
contrasts = (0.05,0.1,0.2,0.4,0.6,0.8,1)
radii = np.linspace(0, stim_size *1.5, 9) + 1


""" break point here in .ipynb file"""


### extract the data and train all the models
### it'll take a lot of time! Be patient!
for group_name, condition, date in zip(group_names, cond_names, dates):
    print(group_name + " " + condition)
    #if extract_data:
        #npc.extract_data(gen_path,group_name,date,condition)
    for layer in layers:
        npc.train_neural_model(gen_path,group_name,condition,layer,layers)

""" break point here in .ipynb file"""

### plot and save the modeled RFs and MEIs:


for group_name, condition in zip(group_names, cond_names):
    print("hello")
    print('======================')
    print('Processing: ' + group_name + ', ' + condition)
    if condition != '':
        condition = '_' + condition
    mouse_model,n_neurons,good_neurons,reds,mean_reliab,snr = npc.load_mouse_model(gen_path,group_name,condition)
    all_corrs = npc.plot_layer_corrs(gen_path,group_name,layers[1:],condition)
    corrs = npc.compute_val_corrs(gen_path,group_name,mouse_model,condition)
    goods,all_good_rfs = npc.good_neurons(mouse_model,n_neurons,corrs,make_plots=False)
    #_ = npc.gaussian_RFs(gen_path,all_good_rfs,goods,corrs,all_corrs,pixperdeg,stim_size,mask_size,mouse_model,group_name,condition)
    #npc.generate_MEIS(gen_path,mouse_model,group_name,goods,condition)    
    npc.generate_single_MEIS(gen_path,all_corrs,group_name,n_neurons,layers[1:],goods,condition)    
    #npc.generate_MEIS(gen_path,mouse_model,group_name,goods,condition,-1)    
    #npc.generate_surround_MEIS(gen_path,mouse_model,group_name,goods,condition)
    #if extra_stims:
        #npc.get_full_resps(gen_path,mouse_model,goods,group_name,condition,pics_folder)
    #if n_neurons > 5:
        #npc.ori_tuning(gen_path,stim_size,wavelengths,orientations,phases,contrasts,mouse_model,group_name,goods,condition)
        #npc.size_tuning(gen_path,stim_size,radii,mouse_model,group_name,goods,condition)
        #npc.output(gen_path,group_name,condition,goods,good_neurons,reds,mean_reliab,snr,resps=False)
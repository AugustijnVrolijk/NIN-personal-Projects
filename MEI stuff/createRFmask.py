import pandas as pd
import numpy as np
import os
import cv2
import shutil
import matplotlib.pyplot as plt

from concurrent.futures import ThreadPoolExecutor
from dataprep import getMinSpikesAndFilter
from pathlib import Path
from imageComp import npImage, expand_folder_path
from tqdm import tqdm
from PIL import Image
from ImageAnalysis import smallest_common_divisor_above_threshold, compute_entropy_map
from skimage.util import view_as_windows
from skimage.measure import shannon_entropy

def conv_coords(azi, ele):
    """
    image coords are:
    0,0      1920,0
    0,1080   1920,1080

    azi,ele coords are:

    -60,45   60,45
    -60,-45  60,-45
    """
    #normalise to 0-120, 0-90
    azi += 60 #    0,90 120,90
    ele += 45 #    0,0  120,0

    #invert ele so its similar to image coords
    ele = 90 - ele
    """
    0,0   120,0
    0,90  120,90
    """  
    #1920 % 120 = 
    #1080 % 90 = 12
    x_coord = azi*16
    y_coord = ele*12
    return int(x_coord), int(y_coord)

def conv_radius(FWHM):
    radius = FWHM/2
    x_rad = radius*16
    y_rad = radius*12
    return int(x_rad), int(y_rad)

def get_avg_condition_rf(sourceDir):
    conditions = ["FamiliarNotOccluded","FamiliarOccluded","NovelNotOccluded","NovelOccluded"]
    targetDir = Path(os.path.join(sourceDir, "avg_rf_condition"))
    targetDir.mkdir(exist_ok=True)

    for cond in conditions:
        curDir = Path(os.path.join(sourceDir, cond))
        paths = [f for f in curDir.iterdir() if f.is_file() and f.name != "Thumbs.db"]
        avg = npImage(average_RF(paths))
        avg.save(os.path.join(targetDir, f"{cond}.png"))

def average_RF(paths:list[str]) -> np.ndarray:
   
    img = Image.open(paths[0]).convert('L')  # Convert to 8-bit grayscale
    total_matrix = np.array(img).astype(np.float64)

    for val in tqdm(paths[1:]):
        img = Image.open(val).convert('L')  # Convert to 8-bit grayscale

        total_matrix += np.array(img).astype(np.float64)

    total_matrix /= len(paths)
    return total_matrix

def mergeCoordInfo(dirName, allInfo, filtered):
    info = pd.read_csv(allInfo)
    filtered = pd.read_csv(filtered)

    filtered_renamed = filtered.rename(columns={
    'mouseNeuron': 'Neuron',
    "respFamiliarNO": "FamiliarNO",	
    "respFamiliarO": "FamiliarO", 
    "respNovelNO": "NovelNO",	
    "respNovelO": "NovelO",
    })
    merged_df = filtered_renamed.merge(info, on=['Mouse', 'Neuron'], how="left")

    savePath = os.path.join(dirName, "filtered_with_coords_info.csv")
    to_drop = ["RFscore", "respFamiliarNO", "respFamiliarO", "respNovelNO", "respNovelO"]
    merged_df.drop(to_drop, axis=1,inplace=True)
    merged_df.to_csv(savePath, index=False)
    
def filterCoordInfo(filtered_all_info_raw):
    def selCondition(row):
        on_ok = row['onRSQ'] > 0.33 and row['onSNR'] > 4
        off_ok = row['offRSQ'] > 0.33 and row['offSNR'] > 4

        if on_ok:
            return 'on'
        elif off_ok:
            return 'off'
        else:
            raise ValueError(f"{row}\nNeither on or off was good, this shouldn't happen")  # Neither condition is good

    filtered_all_info = pd.read_csv(filtered_all_info_raw)
    filtered_all_info['keep'] = filtered_all_info.apply(selCondition, axis=1)
    
    for param in ['Azi', 'Ele', 'FWHM']:  # Add more as needed
        filtered_all_info[param] = filtered_all_info.apply(lambda row: row[f"{row['keep']}{param}"], axis=1)

    # Drop old columns
    drop_cols = [f"{cond}{param}" for cond in ['on', 'off'] for param in ['Azi', 'Ele', 'RSQ', 'SNR', 'FWHM']]
    filtered_all_info.drop(columns=drop_cols + ['keep'], axis=1, inplace=True)

    flags = ["FamiliarNO","FamiliarO","NovelNO", "NovelO"]
    for flag in flags:
        temp = filtered_all_info[filtered_all_info[flag] == True].copy()
        temp.drop(flags,axis=1, inplace=True)
        savePath = os.path.join(Path(filtered_all_info_raw).parent, f"{flag}.csv")
        temp.to_csv(savePath, index=False) 


def plot_totalRF(imgBase, dataBase, imgConds, dataConds):
        
    for i, cond in enumerate(imgConds):
        dataPath = os.path.join(dataBase, f"{dataConds[i]}.csv")
        imgPath = os.path.join(imgBase, f"{cond}.png")
        df = pd.read_csv(dataPath)  # with 'x', 'y', 'z' columns

        # Load grayscale image and convert to BGR to draw colored circles
        gray = cv2.imread(imgPath, cv2.IMREAD_GRAYSCALE)
        image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        # Draw circles
        for _, row in df.iterrows():
            center = (conv_coords(row['Azi'], row['Ele']))
            radius = (conv_radius(row['FWHM']))  # or use a fixed value like 5 if z is not radius
            color = (0, 255, 0)  # green in BGR
            thickness = 2
            cv2.ellipse(image, center, radius, 
                        angle=0,
                        startAngle=0,
                        endAngle=360,
                        color=color, thickness=thickness)

        # Save result
        savePath = os.path.join(imgBase, f"{cond}_drawn.png")
        cv2.imwrite(savePath, image)

def find_totalRF_contour(imgBase, dataBase, imgConds, dataConds, save=False):
    
    retDict = {}
    for i, cond in enumerate(imgConds):
        dataPath = os.path.join(dataBase, f"{dataConds[i]}.csv")
        imgPath = os.path.join(imgBase, f"{cond}.png")
        df = pd.read_csv(dataPath)  # with 'x', 'y', 'z' columns

        # Load grayscale image and convert to BGR to draw colored circles
        Image = cv2.imread(imgPath, cv2.IMREAD_GRAYSCALE)
        mask = np.zeros_like(Image[:, :])

        # Draw circles
        for _, row in df.iterrows():
            center = (conv_coords(row['Azi'], row['Ele']))
            radius = (conv_radius(row['FWHM']))  # or use a fixed value like 5 if z is not radius
            cv2.ellipse(mask, center, radius, 
                        angle=0,startAngle=0,endAngle=360,
                        color=255, thickness=-1)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        assert len(contours) == 1, f"more than one contour found for {cond}, disjointed ROIs"
        cv2.drawContours(Image, [contours[0]], -1, 255, 2)

        mask = np.zeros_like(Image[:, :])
        cv2.drawContours(mask, [contours[0]], -1, 255, -1) #filled mask 0 if not contour, 255 if contour
        if save:
            savePath = os.path.join(imgBase, f"{cond}_merged_contour.png")
            cv2.imwrite(savePath, Image)
        retDict[cond] = mask
    return retDict

@expand_folder_path
def masked_mean_calc(images, mask):
    total = len(images)
    masked_mean = np.ndarray(total)
    masked_sum_mean = np.ndarray(total)
    for i, path in enumerate(images):
        #print(f"Processing {i+1}/{total}: {path}")
        im_arr = cv2.imread(path)
        masked_pixels = im_arr[mask == 255]
        masked_mean[i] = masked_pixels.mean()
        masked_sum_mean[i] = masked_pixels.sum()
    return masked_mean, masked_sum_mean

def findMeanPixelIntensity(contourMasksDict, imgBase):

    for cond, mask in contourMasksDict.items():
        cond_path = os.path.join(imgBase, cond)
        masked_mean, masked_sum_mean = masked_mean_calc(cond_path, mask)
        mask_size = (mask.sum()/255)
        print(f"\n{cond}:")
        print(f"Contour pixel intensity mean:            {masked_mean.mean():.2f} ± {masked_mean.std():.2f}")
        print(f"Contour pixel intensity sum mean:        {masked_sum_mean.mean():.2f} ± {masked_sum_mean.std():.2f}")
        print(f"Contour area (n pixels):                 {mask_size}")
        print(f"(Contour pixel intensity sum mean)/area: {masked_sum_mean.mean()/mask_size}")

def cmp_filtered_neurons():
    #getMinSpikesAndFilter(csv_path, activationsPath, spikeThreshold:int=0.1, **kwargs):
    #filterDataSheet(rawData=None, saveFolder:str=r"C:\Users\augus\NIN_Stuff\data\koenData\RFanalysis",
    #                enforce:bool=False, SNRThresh:float=4, RSQThresh:float=0.33, nSpikesThresh:int=50,
    #                RFCheck:bool=True, ResponseThresh:float=0.5, saveName:str="passedFilter.csv", ignoreRSQSNR:bool=True)
    
    info_path = r"C:\Users\augus\NIN_Stuff\data\koenData\newRFQuant\info.csv"
    activations = r"C:\Users\augus\NIN_Stuff\data\koenData\newRFQuant\NeuronActivations.npy"
    saveFolder = r"C:\Users\augus\NIN_Stuff\data\koenData\newRFQuant"

    #saveName1 = "passedMinSpikes.csv"
    #orig = getMinSpikesAndFilter(info_path, activations, spikeThreshold=0.09, nSpikesThresh=25, saveFolder=saveFolder,saveName=saveName1)
    #saveName2 = "passedAllFilter.csv"
    #filtered = getMinSpikesAndFilter(info_path, activations, spikeThreshold=0.09, nSpikesThresh=25, saveFolder=saveFolder,saveName=saveName2,ignoreRSQSNR = False)
    rf_orig = r"C:\Users\augus\NIN_Stuff\data\koenData\RFSigNormal"
    passed_filter = r"C:\Users\augus\NIN_Stuff\data\koenData\newRFQuant\passedAllFilter.csv"
    #copy_filtered_neurons(rf_orig, saveFolder, passed_filter)
    raw_imgs = r"C:\Users\augus\NIN_Stuff\data\koenData\muckli4000npy"
    filtered_all_info = r"C:\Users\augus\NIN_Stuff\data\koenData\newRFQuant\filtered_with_coords_info.csv"
    #filterCoordInfo(filtered_all_info)
    data1 = r"C:\Users\augus\NIN_Stuff\data\koenData\newRFQuant\FamiliarNO.csv"
    imgBase = r"c:\Users\augus\NIN_Stuff\data\koenData\newRFQuant\avg_rf_condition"
    imgConds = ["FamiliarNotOccluded","FamiliarOccluded","NovelNotOccluded","NovelOccluded"]
    dataConds = ["FamiliarNO","FamiliarO","NovelNO","NovelO"]
    contourDict = find_totalRF_contour(imgBase,saveFolder,imgConds, dataConds)
    findMeanPixelIntensity(contourDict, saveFolder)
    pass

if __name__ == "__main__":
    pass
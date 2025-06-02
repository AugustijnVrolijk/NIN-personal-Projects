import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
import cv2

from skimage import measure
from tqdm import tqdm
from PIL import Image
from pathlib import Path
from matLoader import loadMat
from imageComp import opti_weighted_average_images, npImage, expand_folder_path
from functools import wraps
from ImageAnalysis import analyze_image_folder, checkName


def normaliseNeuron(arr:np.ndarray, idx:int=None):
    #assume format is #trials , neurons

    if isinstance(idx, int):
        curArr = arr[:, idx]
    else:
        curArr = arr

    min_val = np.min(curArr)
    max_val = np.max(curArr)

    normalized_array = (curArr - min_val) / (max_val - min_val)
    return normalized_array

def getPaths(basePath:str, log:np.ndarray, extension:str=".npy"):
    if log.ndim != 1:
        raise ValueError("expect one dimensional path")
    length = log.shape[0]

    paths = [0]*length
    for i in range(length):
        end = f"{int(log[i]):04d}{extension}"
        origPath = os.path.join(basePath, end)
        corPath = npImage._checkPath(origPath, extension)
        paths[i] = corPath
    return paths

def saveResMatImg(arr:np.ndarray, savePath:str, bufferColour:str="black"):
    imgShape = arr.shape
    assert len(imgShape) == 3
    assert imgShape[2] == 2
    #3 dims, 12, 16, 2
    #virtually stack them into 25*16 img and then save that img


    cleanArr = npImage._normalise(arr, 255).astype(np.uint8)

    if bufferColour.lower().strip() == "black":
        bufferInt = 0
    elif bufferColour.lower().strip() == "white":
        bufferInt = 255
    else:
        raise ValueError("Unknown buffer colour, accepted are: 'black', 'white'")

    buffer = np.full((2, imgShape[1]), bufferInt)
    mergedImg = np.vstack([cleanArr[:, :, 0], buffer, cleanArr[:, :, 1]])
    mergedImg = npImage(mergedImg)
    mergedImg.resize(15, resample=Image.NEAREST)
    mergedImg.save(savePath)
    return


def getRF(trimmedActivations, neuron, imgIDPaths):
    normalised = normaliseNeuron(trimmedActivations, neuron)
    receptive_field = npImage(opti_weighted_average_images(imgIDPaths, normalised))

    #post processing to make the image more visible
    receptive_field.blur(3, save=True)
    receptive_field.gamma_correction(2, save=True)
    return receptive_field

def calcRFs(micePath, inputCSV, saveFolder, baseIMGPath, label:str="",overwrite=False):
    signalToUse = "CaDec"
    
    resMatPath = r"C:\Users\augus\NIN_Stuff\data\koenData\Koen_to_Augustijn\RFByMiceData.mat"
    extension = ".png"

    def fetchMiceData(mouse):
        print(f"loading {mouse} data")
        mouseData = loadMat(micePath[mouse])
        print(f"loaded {mouse} data")
        signal = getNeuronActivations(mouseData.Res[signalToUse], cutBefore=9, cutAfter=22)

        imgID = mouseData.info.Stim.Log
        imgIDPaths = getPaths(baseIMGPath, imgID)
        for path in imgIDPaths:
            assert os.path.isfile(path)

        return signal, imgIDPaths
        
    SNRdata = loadMat(resMatPath)

    GoodNeurons = pd.read_csv(inputCSV)

    curMouse = GoodNeurons.loc[0, "Mouse"]
    signal, imgIDPaths = fetchMiceData(curMouse)
    try:
        trueRF = True
        resMat = SNRdata[curMouse].info.resMat  #shape like: (neurons, 12, 16, 2)
    except KeyError:
        trueRF = False

    totalLen = len(GoodNeurons)
    for i, row in GoodNeurons.iterrows():
        print(f"processing: neuron {i}/{totalLen}")
        #check if we need to load in a different mouse's data
        if not curMouse == row['Mouse']:
            curMouse = row['Mouse']
            signal, imgIDPaths = fetchMiceData(curMouse)
            try:
                trueRF = True
                resMat = SNRdata[curMouse].info.resMat  #shape like: (neurons, 12, 16, 2)
            except KeyError:
                trueRF = False

        #check if this image has already been processed and saved
        FamiliarNO, FamiliarO, NovelNO, NovelO = row[['respFamiliarNO', 'respFamiliarO', 'respNovelNO', 'respNovelO']]

        saveDir = getDirName(saveFolder, FamiliarNO, FamiliarO, NovelNO, NovelO)
        saveName = f"{getFileName(FamiliarNO, FamiliarO, NovelNO, NovelO)}_{curMouse}_{row['mouseNeuron']}"
        savePath = Path(os.path.join(saveDir, saveName))
        checkSaveName = f"{saveName}{extension}"
        checkSavePath = Path(os.path.join(saveDir, checkSaveName))
        saveResName = f"{saveName}_true{extension}"
        saveResPath = Path(os.path.join(saveDir, saveResName))

        if checkSavePath.exists() and not overwrite:
            continue
       
        saveName = f"{getFileName(FamiliarNO, FamiliarO, NovelNO, NovelO)}_{curMouse}_{row['mouseNeuron']}{label}"
        #get receptive field
        receptive_field = getRF(signal, row['mouseNeuron'], imgIDPaths)

        receptive_field.save(savePath, extension=extension)
        if trueRF:
            saveResMatImg(resMat[row['mouseNeuron'],:,:,:],saveResPath,"white")

def cleanRFs(inputCSV, inputCSV_all, saveFolder, label:str=""):
    extension = ".png"
    GoodNeurons = pd.read_csv(inputCSV)
    GoodNeurons_all = pd.read_csv(inputCSV_all)

    curMouse = GoodNeurons.loc[0, "Mouse"]

    totalLen = len(GoodNeurons_all)
    row_i = 0
    for i, row_all in GoodNeurons_all.iterrows():
        print(f"processing: neuron {i}/{totalLen}")
        #check if we need to load in a different mouse's data
        if not curMouse == row_all['Mouse']:
            curMouse = row_all['Mouse']

        isDuplicate = False

        row = GoodNeurons.iloc[row_i]
        if (row['Mouse'] == row_all['Mouse']) and (row['mouseNeuron'] == row_all['mouseNeuron']):
            isDuplicate = True
            t1, t2, t3, t4  = row_all[['respFamiliarNO', 'respFamiliarO', 'respNovelNO', 'respNovelO']]
            l1, l2, l3, l4 = row[['respFamiliarNO', 'respFamiliarO', 'respNovelNO', 'respNovelO']]
            assert (t1 == l1) and (t2 == l2) and (t3 == l3) and (t4 == l4)
            row_i += 1
       
        FamiliarNO, FamiliarO, NovelNO, NovelO = row_all[['respFamiliarNO', 'respFamiliarO', 'respNovelNO', 'respNovelO']]
        saveDir = getDirName(saveFolder, FamiliarNO, FamiliarO, NovelNO, NovelO)
        saveNameCore = f"{getFileName(FamiliarNO, FamiliarO, NovelNO, NovelO)}_{curMouse}_{row_all['mouseNeuron']}"

        true = f"{saveNameCore}{extension}"
        truePath = Path(os.path.join(saveDir, true))
        trueResName = f"{saveNameCore}_true{extension}"
        trueResPath = Path(os.path.join(saveDir, trueResName))

        duplicate = f"{saveNameCore}{label}{extension}"
        duplicatePath = Path(os.path.join(saveDir, duplicate))
        duplicateResName = f"{saveNameCore}{label}_true{extension}"
        duplicateResPath = Path(os.path.join(saveDir, duplicateResName))

        if isDuplicate:
            duplicatePath.unlink(True)
            duplicateResPath.unlink(True) 
        else:
            truePath.unlink(True)
            trueResPath.unlink(True)

def buildActivationMatrix():
    return

def getDirName(base:Path | str, FamiliarNO:bool, FamiliarO:bool, NovelNO:bool, NovelO:bool) -> Path:
    NotOccluded = False
    occluded = False
    
    if FamiliarNO or NovelNO:
        NotOccluded = True
    if FamiliarO or NovelO:
        occluded = True

    if occluded and NotOccluded:
        subFolder = "both"
    elif occluded:
        subFolder = "occluded"
    elif NotOccluded:
        subFolder = "notOccluded"
    else:
        subFolder = ""

    saveDir = Path(os.path.join(base, subFolder))
    saveDir.mkdir(exist_ok=True)
    return saveDir

def getFileName(FamiliarNO:bool, FamiliarO:bool, NovelNO:bool, NovelO:bool):
    
    familiar = False
    novel = False

    if FamiliarNO or FamiliarO:
        familiar = True
    if NovelNO or NovelO:
        novel = True

    if familiar and novel:
        nameType = "both"
    elif familiar:
        nameType = "familiar"
    elif novel:
        nameType = "novel"
    else:
        raise ValueError("neuron doesn't belong to any condition")

    return nameType

def perform_analysis(dest, name_skip, label):
    both = r"C:\Users\augus\NIN_Stuff\data\koenData\Koen_to_Augustijn\RFbyResponseTypeFull\both"
    nOcc = r"C:\Users\augus\NIN_Stuff\data\koenData\Koen_to_Augustijn\RFbyResponseTypeFull\notOccluded"
    occ = r"C:\Users\augus\NIN_Stuff\data\koenData\Koen_to_Augustijn\RFbyResponseTypeFull\occluded"
    print("analysing both")
    labelB = f"_both{label}"
    analyze_image_folder(both, dest, label=labelB, name_skip=name_skip, patch_size=30, resize=False)
    print("analysing notOccluded")
    labelNO = f"_notOccluded{label}"
    analyze_image_folder(nOcc, dest, label=labelNO, name_skip=name_skip, patch_size=30, resize=False)
    print("analysing occluded")
    labelO = f"_occluded{label}"
    analyze_image_folder(occ, dest, label=labelO, name_skip=name_skip, patch_size=30, resize=False)

def blur_image(origin:str, dest:str, weight:float=0.2, sigma:float=10) -> npImage:
    img = npImage(origin)

    img.blur(sigma=sigma)
    img.tv_smoothing(weight=weight)
    img.save(dest)
    return

def blur_task(args):
    return blur_image(*args)

@expand_folder_path
def blur_folder(paths:str, skip:list, dest, weight:float=0.2, sigma:float=15, label:str="_blurred"):
    tasks = []
    for temp_path in paths:
        if checkName(temp_path, skip):
            continue
        origin = Path(temp_path)
        temp_dest = Path(os.path.join(dest, f"{origin.stem}{label}{origin.suffix}"))
        tasks.append((origin, temp_dest, weight, sigma))


    with ProcessPoolExecutor() as executor:
        list(tqdm(executor.map(blur_task, tasks)))

    return

def main_blur():
    n_s1 = ["true", "both", "novel", "noMinSpike"]
    n_s2 = ["true", "both", "familiar", "noMinSpike"]

    p1 = r"C:\Users\augus\NIN_Stuff\data\koenData\Koen_to_Augustijn\RFbyResponseTypeFull\notOccluded"
    dest1 = r"C:\Users\augus\NIN_Stuff\data\koenData\Koen_to_Augustijn\RFanalysis\notOccluded\familiar"
    dest2 = r"C:\Users\augus\NIN_Stuff\data\koenData\Koen_to_Augustijn\RFanalysis\notOccluded\novel"
    blur_folder(p1, n_s1, dest1)
    blur_folder(p1, n_s2, dest2)

    p2 = r"C:\Users\augus\NIN_Stuff\data\koenData\Koen_to_Augustijn\RFbyResponseTypeFull\occluded"
    dest3 = r"C:\Users\augus\NIN_Stuff\data\koenData\Koen_to_Augustijn\RFanalysis\occluded\familiar"
    dest4 = r"C:\Users\augus\NIN_Stuff\data\koenData\Koen_to_Augustijn\RFanalysis\occluded\novel"
    blur_folder(p2, n_s1, dest3)
    blur_folder(p2, n_s2, dest4)

def measureBlobs():
    # Load smoothed image
    path = r"C:\Users\augus\NIN_Stuff\data\koenData\RFanalysis\notOccluded\familiar\familiar_Ajax_306_blurred.png"
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    # Threshold
    binary = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 71, 1)

    # Label blobs
    temp = npImage(binary)
    temp.save(r"C:\Users\augus\NIN_Stuff\data\koenData\RFanalysis\notOccluded\familiar\familiar_Ajax_306_tester.png")
    labels = measure.label(binary)
    props = measure.regionprops(labels, intensity_image=img)
    for i, prop in enumerate(props):
        print(f"Blob {i}: Area={prop.area}, Mean Intensity={prop.mean_intensity:.2f}")
    return

if __name__ == "__main__":
    micePath = { #In alphabetical order for the 3312 neurons
        "Ajax":r"C:\Users\augus\NIN_Stuff\data\koenData\Koen_to_Augustijn\Ajax_20241012_001_normcorr_SPSIG_Res.mat",
        "Anton":r"C:\Users\augus\NIN_Stuff\data\koenData\Koen_to_Augustijn\Anton_20241123_1502_normcorr_SPSIG_Res.mat",
        "Bell":r"C:\Users\augus\NIN_Stuff\data\koenData\Koen_to_Augustijn\Bell_20241122_1150_normcorr_SPSIG_Res.mat",
        "Fctwente":r"C:\Users\augus\NIN_Stuff\data\koenData\Koen_to_Augustijn\Fctwente_20241011_001_normcorr_SPSIG_Res.mat",
        "Feyenoord":r"C:\Users\augus\NIN_Stuff\data\koenData\Koen_to_Augustijn\Feyenoord_20241011_001_normcorr_SPSIG_Res.mat",
        "Jimmy":r"C:\Users\augus\NIN_Stuff\data\koenData\Koen_to_Augustijn\Jimmy_20241123_1129_normcorr_SPSIG_Res.mat",
        "Lana":r"C:\Users\augus\NIN_Stuff\data\koenData\Koen_to_Augustijn\Lana_20241012_001_normcorr_SPSIG_Res.mat",
    }
    inputCSV = r"C:\Users\augus\NIN_Stuff\data\koenData\Koen_to_Augustijn\RFbyResponseTypeFull\lessStringentWithMinSpikes.csv"
    inputCSV_all = r"c:\Users\augus\NIN_Stuff\data\koenData\Koen_to_Augustijn\RFbyResponseTypeFull\lessStringent.csv"
    saveFolder = r"C:\Users\augus\NIN_Stuff\data\koenData\Koen_to_Augustijn\RFbyResponseTypeFull"

    muckli4000 = Path(r"C:\Users\augus\NIN_Stuff\data\koenData\Koen_to_Augustijn\Muckli4000Images")
    images = Path(r"C:\Users\augus\NIN_Stuff\data\koenData\Koen_to_Augustijn\muckli4000npy")

    measureBlobs()

    

def calcCleanRFs():
    calcRFs(micePath, inputCSV, saveFolder,baseIMGPath=images, overwrite=True)
    calcRFs(micePath, inputCSV_all, saveFolder, baseIMGPath=images, label="_noMinSpike")
    cleanRFs(inputCSV, inputCSV_all, saveFolder, label="_noMinSpike")
    filterDataSheet(False, -1, -1, 35, True, 0.5, "lessStringentWithMinSpikes.csv", ignoreRSQSNR=True)

def analysis_bulk():
    dest = r"C:\Users\augus\NIN_Stuff\data\koenData\Koen_to_Augustijn\RFbyResponseTypeFull\analysis"
    dest1 = os.path.join(dest, r"novel")
    name_skip1 = ["true", "noMinSpike", "familiar"]
    name_skip11 = ["true", "familiar"]
    label1 = ""
    
    dest2 = os.path.join(dest, r"familiar")
    name_skip2 = ["true", "noMinSpike", "novel"]
    name_skip21 = ["true", "novel"]
    label2 = "_all"

    name_skip3 = ["true", "noMinSpike"]
    name_skip31 = ["true"]
    perform_analysis(dest1, name_skip1, label1)
    perform_analysis(dest1, name_skip11, label2)

    perform_analysis(dest2, name_skip2, label1)
    perform_analysis(dest2, name_skip21, label2)

    perform_analysis(dest, name_skip3, label1)
    perform_analysis(dest, name_skip31, label2)

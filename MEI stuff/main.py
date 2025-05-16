import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

from PIL import Image
from pathlib import Path
from matLoader import loadMat
from imageComp import weighted_average_images, npImage
from functools import wraps

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

def convertStr2Tuple(func):
    @wraps(func)
    def inner(*args, **kwargs):
        newArgs = []
        for i, arg in enumerate(args):
            if isinstance(arg, str):
                if arg[0] == '[' and arg[-1] == ']':
                    temp = arg.strip("[]")
                    parts = temp.split()
                    arg = tuple(np.nan if p.lower() == 'nan' else float(p) for p in parts)
            newArgs.append(arg)

        for key, arg in kwargs.items():
            if isinstance(arg, str):
                if arg[0] == '[' and arg[-1] == ']':
                    temp = arg.strip("[]")
                    parts = temp.split()
                    kwargs[key] = tuple(np.nan if p.lower() == 'nan' else float(p) for p in parts)
        return func(*newArgs, **kwargs)
    return inner

@convertStr2Tuple
def checkRSQSNR(snr, rsq, enforce:bool=True, SNRthreshold:float=4, RSQthreshold:float=0.33):
    light = False
    if snr[0] > SNRthreshold and rsq[0] > RSQthreshold:
        light = True

    dark = False
    if snr[1] > SNRthreshold and rsq[1] > RSQthreshold:
        dark = True

    if enforce:
        if (light and dark):
            return True
    else:
        if (light or dark):
            return True
        
    return False

def getGoodNeurons(info:dict, enforce:bool=True, SNRthreshold:float=4, RSQthreshold:float=0.33):
    nNeurons = info.SNR.shape[0]
    snr = info.SNR
    rsq = info.RSQ
    score = []
    goodNeurons = []
    for i in range(nNeurons):
        if checkRSQSNR(snr[i, :], rsq[i, :], enforce, SNRthreshold, RSQthreshold):    
           goodNeurons.append(i)

    return goodNeurons

def getNeuronActivations(signal:np.ndarray, cutBefore:int=9, cutAfter:int=-1):
    assert len(signal.shape) == 3

    trimmed = signal[cutBefore:cutAfter, :, :]
    avg = trimmed.sum(axis=0)

    return avg

def collateDataSheet(micePath):
   
    signalToUse = "CaDec"
    spikeThreshold = 0.05

    NOandOScoresPath = r"C:\Users\augus\NIN_Stuff\data\koenData\Koen_to_Augustijn\dataForAugustijn\dataForAugustijn.mat"
    SNRandRSQScorePath = r"C:\Users\augus\NIN_Stuff\data\koenData\Koen_to_Augustijn\RFByMiceData.mat"
    
    
    SNRandRSQScore = loadMat(SNRandRSQScorePath)
    NOandOScores = loadMat(NOandOScoresPath)
    NOandOScores = NOandOScores.data['data']

    """ data(:,1)  all neurons with good RFs that were included in the analysis (2106/3312 neurons)
        data(:,2)  average response of included neurons to familiar NO stimuli
        data(:,3)  average response of included neurons to familiar O stimuli
        data(:,4)  average response of included neurons to novel NO stimuli
        data(:,5)  average response of included neurons to novel O stimuli """
    
    totalN, _ = NOandOScores.shape

    placeHolder = [np.nan] * totalN
    placeHolderInt = [-1] * totalN

    finalData = pd.DataFrame({
        'Mouse': [None] * totalN,
        'Neuron': placeHolderInt,
        'RFscore': NOandOScores[:, 0],
        'RSQscore': placeHolder,
        'SNRscore': placeHolder,
        'nSpikesAboveThreshold': placeHolderInt,
        'respFamiliarNO': NOandOScores[:, 1],
        'respFamiliarO': NOandOScores[:, 2],
        'respNovelNO': NOandOScores[:, 3],
        'respNovelO': NOandOScores[:, 4]
    })
    finalData['RSQscore'] = finalData['RSQscore'].astype(object)
    finalData['SNRscore'] = finalData['SNRscore'].astype(object)

    curN = 0
    for mouse in micePath.keys():
        print(mouse)
        mouseData = loadMat(micePath[mouse])
        print("loaded")
        signal = mouseData.Res[signalToUse]
        frames, trials, mouseNeurons = signal.shape
        assert (frames == 24) and (trials == 4000)
        nSpikes = getNeuronActivations(signal, 9,22)
        #now shaped (4000, neurons)
        nSpikes = normaliseNeuron(nSpikes)
        nSpikes = (nSpikes > spikeThreshold).sum(axis=0)
        #now shaped (neurons)

        miceNames = np.full(mouseNeurons, mouse)
        neurons = np.arange(mouseNeurons)
        try:
            mouseSNR = SNRandRSQScore[mouse].info
            snr = mouseSNR.SNR
            rsq = mouseSNR.RSQ
        except KeyError:
            snr = np.full((mouseNeurons, 2), np.nan)
            rsq = np.full((mouseNeurons, 2), np.nan)

        
        values = np.column_stack((miceNames, neurons, nSpikes))
        cols = finalData.columns.get_indexer(['Mouse', 'Neuron', 'nSpikesAboveThreshold'])
        finalData.iloc[curN:curN+mouseNeurons, cols] = values
        finalData.iloc[curN:curN+mouseNeurons, cols] = values

        for i in range(mouseNeurons):
            #pandas can't seem to assign tuples using iloc or loc, so have to manually assign each cell
            finalData.at[curN+i, 'SNRscore'] = snr[i, :]
            finalData.at[curN+i, 'RSQscore'] = rsq[i, :]

        curN += mouseNeurons

    saveFolder = r"C:\Users\augus\NIN_Stuff\data\koenData\Koen_to_Augustijn\RFbyResponseType"
    savePath = os.path.join(saveFolder, "info.csv")
    finalData.to_csv(savePath, index=False)
    
def filterDataSheet(enforce:bool=True, SNRThresh:float=4, RSQThresh:float=0.33, nSpikesThresh:int=50, RFCheck:bool=True, ResponseThresh:float=0.5, saveName:str="passedFilter.csv", ignoreRSQSNR:bool=False):
    saveFolder = r"C:\Users\augus\NIN_Stuff\data\koenData\Koen_to_Augustijn\RFbyResponseType"
    loadPath = os.path.join(saveFolder, "info.csv")
    rawData = pd.read_csv(loadPath)

    FinalData = pd.DataFrame({"NeuronID":pd.Series(dtype="int"), 
                              "Mouse":pd.Series(dtype="str"),
                              "mouseNeuron":pd.Series(dtype="int"), 
                              'respFamiliarNO':pd.Series(dtype="bool"),
                                'respFamiliarO':pd.Series(dtype="bool"),
                                'respNovelNO':pd.Series(dtype="bool"),
                                'respNovelO':pd.Series(dtype="bool")})
    curLen = 0

    checkResponse = lambda x: 1 if x >= ResponseThresh else 0

    for i, row in rawData.iterrows():
        #check SNR and RSQ
        if (not checkRSQSNR(row['SNRscore'], row['RSQscore'], enforce, SNRThresh, RSQThresh)) and (not ignoreRSQSNR):
            continue 
        #Check RF
        if RFCheck and (row['RFscore'] == 0):
            continue
        #check nSpikes aboveThreshold
        if row['nSpikesAboveThreshold'] < nSpikesThresh:
            continue

        respFamiliarNO = checkResponse(row['respFamiliarNO'])
        respFamiliarO = checkResponse(row['respFamiliarO'])
        respNovelNO  = checkResponse(row['respNovelNO'])
        respNovelO = checkResponse(row['respNovelO'])

        if not (respFamiliarNO or respFamiliarO or respNovelNO or respNovelO):
            continue

        FinalData.loc[curLen] = [i, row['Mouse'], row['Neuron'], respFamiliarNO, respFamiliarO, respNovelNO, respNovelO]
        curLen += 1

    savePath = os.path.join(saveFolder, saveName)
    FinalData.to_csv(savePath, index=False)

def AjaxRFs():
    #stuff to change
    outputDir = Path(r"C:\Users\augus\NIN_Stuff\data\koenData\Koen_to_Augustijn\MEIresults")
    extension = ".png"
    mouseName = "Ajax"
    dataPath = r"C:\Users\augus\NIN_Stuff\data\koenData\Koen_to_Augustijn\Ajax_20241012_001_normcorr_SPSIG_Res.mat"
    spike_threshold = 0.05
    RFPath = r"C:\Users\augus\NIN_Stuff\data\koenData\Koen_to_Augustijn\RFByMiceData.mat"


    fdata = loadMat(dataPath)

    #get signal and clean
    Sig = fdata.Res.CaDec
    trimmedActivations = getNeuronActivations(Sig, cutAfter=19)

    #sort img path names in order of the stimulation log
    imgID = fdata.info.Stim.Log
    baseIMGPath = Path(r"C:\Users\augus\NIN_Stuff\data\koenData\Koen_to_Augustijn\ImagesSmallNPY")
    imgIDPaths = getPaths(baseIMGPath, imgID)
    for path in imgIDPaths:
        assert os.path.isfile(path)

    #getting resmat for true receptive field and good neurons from SNR and RSQ
    SNRdata = loadMat(RFPath)
    goodNeurons = getGoodNeurons(SNRdata[mouseName].info)
    resMat = SNRdata[mouseName].info.resMat  #shape like: (neurons, 12, 16, 2)

    #to log instances over spike threshold
    sums = np.empty(len(goodNeurons))

    for i, neuron in enumerate(goodNeurons):
        print(f"processing neuron: {i}/{len(goodNeurons)}")

        #Calculating RF's via reverse correlation
        normalised = normaliseNeuron(trimmedActivations, neuron)
        sums[i] = np.sum(normalised > spike_threshold)
        receptive_field = npImage(weighted_average_images(imgIDPaths, normalised))

        #post processing to make the image more visible
        receptive_field.blur(3, save=True)
        receptive_field.gamma_correction(2, save=True)
        savePath = os.path.join(outputDir, f"RF_{mouseName}_{neuron}{extension}")
        receptive_field.save(savePath)

        #get resMat for the same image
        saveResPath = os.path.join(outputDir, f"RF_{mouseName}_{neuron}_true{extension}")
        saveResMatImg(resMat[i,:,:,:],saveResPath,"white")

    finalData = pd.DataFrame({
        'neurons':goodNeurons,
        'spikesAboveThreshold':sums
    })
    savePath = os.path.join(outputDir, "info.csv")
    finalData.to_csv(savePath, index=False)    


def getRF(trimmedActivations, neuron, imgIDPaths):
    normalised = normaliseNeuron(trimmedActivations, neuron)
    receptive_field = npImage(weighted_average_images(imgIDPaths, normalised))

    #post processing to make the image more visible
    receptive_field.blur(3, save=True)
    receptive_field.gamma_correction(2, save=True)
    return receptive_field

def calcRFs(micePath, inputCSV, overwrite=False):
    signalToUse = "CaDec"
    saveFolder = r"C:\Users\augus\NIN_Stuff\data\koenData\Koen_to_Augustijn\RFbyResponseType"
    baseIMGPath = Path(r"C:\Users\augus\NIN_Stuff\data\koenData\Koen_to_Augustijn\ImagesSmallNPY")
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
    resMat = SNRdata[curMouse].info.resMat  #shape like: (neurons, 12, 16, 2)

    totalLen = len(GoodNeurons)
    for i, row in GoodNeurons.iterrows():
        print(f"processing: neuron {i}/{totalLen}")
        #check if we need to load in a different mouse's data
        if not curMouse == row['Mouse']:
            curMouse = row['Mouse']
            signal, imgIDPaths = fetchMiceData(curMouse)
            resMat = SNRdata[curMouse].info.resMat  #shape like: (neurons, 12, 16, 2)

        #check if this image has already been processed and saved
        FamiliarNO, FamiliarO, NovelNO, NovelO = row[['respFamiliarNO', 'respFamiliarO', 'respNovelNO', 'respNovelO']]

        saveDir = getDirName(saveFolder, FamiliarNO, FamiliarO, NovelNO, NovelO)
        saveName = f"{getFileName(FamiliarNO, FamiliarO, NovelNO, NovelO)}_{curMouse}_{row['mouseNeuron']}"
        savePath = Path(os.path.join(saveDir, saveName))
        saveResName = f"{saveName}_true{extension}"
        saveResPath = Path(os.path.join(saveDir, saveResName))
        if savePath.exists() and not overwrite:
            if not saveResPath.exists():
                saveResMatImg(resMat[row['mouseNeuron'],:,:,:],saveResPath,"white")
            continue

        #get receptive field
        receptive_field = getRF(signal, row['mouseNeuron'], imgIDPaths)

        receptive_field.save(savePath, extension=extension)
        saveResMatImg(resMat[row['mouseNeuron'],:,:,:],saveResPath,"white")

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
    inputCSV = r"C:\Users\augus\NIN_Stuff\data\koenData\Koen_to_Augustijn\RFbyResponseType\fromFctwente.csv"
    
    #calcRFs(micePath, inputCSV)
    filterDataSheet(False, -1, -1, 35, True, 0.5, "lessStringentWithMinSpikes.csv", ignoreRSQSNR=True)
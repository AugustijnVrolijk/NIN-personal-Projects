import pandas as pd
import numpy as np
import os

from imageComp import npImage
from pathlib import Path
from matLoader import loadMat
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

def collateDataSheet(micePath, baseIMGPath, saveFolder):
   
    signalToUse = "CaSig"

    NOandOScoresPath = r"C:\Users\augus\NIN_Stuff\data\koenData\dataForAugustijn\dataForAugustijn.mat"
    SNRandRSQScorePath = r"C:\Users\augus\NIN_Stuff\data\koenData\RFByMiceData.mat"
    
    
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
        'respFamiliarNO': NOandOScores[:, 1],
        'respFamiliarO': NOandOScores[:, 2],
        'respNovelNO': NOandOScores[:, 3],
        'respNovelO': NOandOScores[:, 4]
    })
    finalData['RSQscore'] = finalData['RSQscore'].astype(object)
    finalData['SNRscore'] = finalData['SNRscore'].astype(object)
    prev_imgIDPaths = None
    final_arr = None
    curN = 0
    for mouse in micePath.keys():

        signal, imgIDPaths = fetchMiceData(mouse, baseIMGPath ,micePath,signalToUse)
        if prev_imgIDPaths is None:
            prev_imgIDPaths = imgIDPaths
        else:
            assert prev_imgIDPaths == imgIDPaths
        
        if final_arr is None:
            final_arr = signal
        else:
            final_arr = np.concatenate((final_arr, signal), axis=1)

        trials, mouseNeurons = signal.shape
        assert trials == 4000

        miceNames = np.full(mouseNeurons, mouse)
        neurons = np.arange(mouseNeurons)
        try:
            mouseSNR = SNRandRSQScore[mouse].info
            snr = mouseSNR.SNR
            rsq = mouseSNR.RSQ
        except KeyError:
            snr = np.full((mouseNeurons, 2), np.nan)
            rsq = np.full((mouseNeurons, 2), np.nan)

        
        values = np.column_stack((miceNames, neurons))
        cols = finalData.columns.get_indexer(['Mouse', 'Neuron'])
        finalData.iloc[curN:curN+mouseNeurons, cols] = values
        finalData.iloc[curN:curN+mouseNeurons, cols] = values

        for i in range(mouseNeurons):
            #pandas can't seem to assign tuples using iloc or loc, so have to manually assign each cell
            finalData.at[curN+i, 'SNRscore'] = snr[i, :]
            finalData.at[curN+i, 'RSQscore'] = rsq[i, :]

        curN += mouseNeurons

    saveFolder = Path(saveFolder)
    saveFolder.mkdir(parents=True, exist_ok=True)
    infoPath = os.path.join(saveFolder, r"info.csv")
    finalData.to_csv(infoPath, index=False)
    neuronActivationsPath = os.path.join(saveFolder, r"normalisedNeuronActivations")
    np.save(neuronActivationsPath, final_arr)
    np.save(os.path.join(saveFolder, r"imagesInTrialOrder"), prev_imgIDPaths)

    return infoPath, neuronActivationsPath

def getMinSpikesAndFilter(csv_path, activationsPath, spikeThreshold:int=0.1, **kwargs):
    rawData = pd.read_csv(csv_path)
    nSpikes_raw = np.load(activationsPath)
    normalised_arr = normaliseNeuron(nSpikes_raw)

    #now shaped (4000, neurons)
    a, b = normalised_arr.shape
    if a == 4000:
        t = 0
    elif b == 4000:
        t = 1

    nSpikes = (normalised_arr > spikeThreshold).sum(axis=t)
    rawData['nSpikesAboveThreshold'] = nSpikes
    filterDataSheet(rawData, **kwargs)

def filterDataSheet(rawData=None, saveFolder:str=r"C:\Users\augus\NIN_Stuff\data\koenData\RFanalysis",
                     enforce:bool=True, SNRThresh:float=4, RSQThresh:float=0.33, nSpikesThresh:int=50, RFCheck:bool=True, ResponseThresh:float=0.5, saveName:str="passedFilter.csv", ignoreRSQSNR:bool=True):
    
    if rawData is None:
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

def fetchMiceData(mouse, baseIMGPath, micePath, signalToUse="CaDec"):
    print(f"loading {mouse} data")
    mouseData = loadMat(micePath[mouse])
    print(f"loaded {mouse} data")
    signal = getNeuronActivations(mouseData.Res[signalToUse], cutBefore=9, cutAfter=-1)

    imgID = mouseData.info.Stim.Log
    imgIDPaths = getPaths(baseIMGPath, imgID)
    for path in imgIDPaths:
        assert os.path.isfile(path)

    return signal, imgIDPaths

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

def getAllNeuronActivations(activationsPath, inputCSV, dest):
    GoodNeurons = pd.read_csv(inputCSV)
    neuronActivations = np.load(activationsPath)

    neuronIDs = GoodNeurons["NeuronID"]

    a, b = neuronActivations.shape
    if a == 4000:
        finalActivations = neuronActivations[:, neuronIDs]
    elif b== 4000:
        finalActivations = neuronActivations[neuronIDs, :]

    np.save(os.path.join(dest, r"FilteredNeuronActivations"), finalActivations)
    return

if __name__ == "__main__":
    micePath = { #In alphabetical order for the 3312 neurons
        "Ajax":r"C:\Users\augus\NIN_Stuff\data\koenData\Ajax_20241012_001_normcorr_SPSIG_Res.mat",
        "Anton":r"C:\Users\augus\NIN_Stuff\data\koenData\Anton_20241123_1502_normcorr_SPSIG_Res.mat",
        "Bell":r"C:\Users\augus\NIN_Stuff\data\koenData\Bell_20241122_1150_normcorr_SPSIG_Res.mat",
        "Fctwente":r"C:\Users\augus\NIN_Stuff\data\koenData\Fctwente_20241011_001_normcorr_SPSIG_Res.mat",
        "Feyenoord":r"C:\Users\augus\NIN_Stuff\data\koenData\Feyenoord_20241011_001_normcorr_SPSIG_Res.mat",
        "Jimmy":r"C:\Users\augus\NIN_Stuff\data\koenData\Jimmy_20241123_1129_normcorr_SPSIG_Res.mat",
        "Lana":r"C:\Users\augus\NIN_Stuff\data\koenData\Lana_20241012_001_normcorr_SPSIG_Res.mat",
    }
    inputCSV = r"C:\Users\augus\NIN_Stuff\data\koenData\RFbyResponseTypeFull\lessStringentWithMinSpikes.csv"
    inputCSV_all = r"c:\Users\augus\NIN_Stuff\data\koenData\RFbyResponseTypeFull\lessStringent.csv"
    saveFolder = r"C:\Users\augus\NIN_Stuff\data\koenData\RFbyResponseTypeFull"
    infoPath = r"C:\Users\augus\NIN_Stuff\data\koenData\RFbyResponseType\info.csv"

    muckli4000 = Path(r"C:\Users\augus\NIN_Stuff\data\koenData\Muckli4000Images")
    imagesInOrderPath = r"C:\Users\augus\NIN_Stuff\data\koenData\RFbyResponseTypeFull\imagesInTrialOrder.npy"
    images = Path(r"C:\Users\augus\NIN_Stuff\data\koenData\muckli4000npy")
    
    sigPath = r"C:\Users\augus\NIN_Stuff\data\koenData\RFSig"
    #collateDataSheet(micePath, images, r"C:\Users\augus\NIN_Stuff\data\koenData\RFSig")
    
    #filterDataSheet(False, -1, -1, 35, True, 0.5, "lessStringentWithMinSpikes.csv", ignoreRSQSNR=True)
    #infoPath, neuronActivationsPath = collateDataSheet(micePath,images)

    info = r"C:\Users\augus\NIN_Stuff\data\koenData\RFSig\info.csv"
    SigActivationsPath = r"C:\Users\augus\NIN_Stuff\data\koenData\RFSig\normalisedNeuronActivations.npy"
    #getMinSpikesAndFilter(info, SigActivationsPath, 0.09, nSpikesThresh=25, saveFolder=sigPath)
    passedFilter = r"C:\Users\augus\NIN_Stuff\data\koenData\RFSig\passedFilter.csv"

    getAllNeuronActivations(SigActivationsPath,passedFilter,sigPath)
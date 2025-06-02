import pandas as pd
import numpy as np
import os

from matLoader import loadMat
from functools import wraps

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
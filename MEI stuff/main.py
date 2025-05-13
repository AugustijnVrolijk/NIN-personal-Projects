import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

from PIL import Image
from pathlib import Path
from matLoader import loadMat
from imageComp import weighted_average_images, npImage


def normaliseNeuron(arr:np.ndarray, idx:int):
    #assume format is #trials , neurons

    curArr = arr[:, idx]

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

def getGoodNeurons(info:dict, enforce:bool=True, SNRthreshold:float=4, RSQthreshold:float=0.33):
    nNeurons = info.SNR.shape[0]
    snr = info.SNR
    rsq = info.RSQ
    score = []
    goodNeurons = []
    for i in range(nNeurons):

        light = False
        if snr[i, 0] > SNRthreshold and rsq[i, 0] > RSQthreshold:
            light = True

        dark = False
        if snr[i, 1] > SNRthreshold and rsq[i, 1] > RSQthreshold:
            dark = True

        if enforce:
            if not (light and dark):
                continue
        else:
            if not (light or dark):
                continue

        goodNeurons.append(i)
        if light:
            if dark:
                tscore = 2
            else:
                tscore = 1
        else:
            tscore = 0
        score.append(tscore)
    
    """ score is graded:
        0 = only passed to dark
        1 = only passed to light
        2 = passed both
    """
    return goodNeurons, score

def getNeuronActivations(signal:np.ndarray, cutBefore:int=9, cutAfter:int=-1):
    assert len(signal.shape) == 3

    trimmed = signal[cutBefore:cutAfter, :, :]
    avg = trimmed.sum(axis=0)

    return avg

def main():
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
    goodNeurons, _ = getGoodNeurons(SNRdata[mouseName].info)
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


if __name__ == "__main__":
    main()
import numpy as np
import torch
import matplotlib.pyplot as plt

def printTLData(data):
    #format is time, trials, neuron
    _, trials, neurons = data.shape
    for i in range(neurons):
        print(f"Neuron {i}:")
        for j in range(trials):
            print(f"trial {j}: {data[:, j, i]}")
    print(f"\n")
    return "\n"

def testSNR(matrix, gt, verbose=False):
    if verbose:
        print("\n---------------- testSNR -----------------\n")
        printTLData(matrix)
        print(f"gt: {gt}")  
        printTLData(matrix[gt, :, :])
        #length, trials, neurons
    baseNoSqueeze = np.nanstd(matrix[gt, :, :], axis=(0,1))
    #standard deviation along all trials and some times for each neuron    

    base = np.squeeze(baseNoSqueeze)
    signalnoSqueezenoMax = np.nanmean(matrix, axis=1)
    #2 dimensional array, times, neurons, gets mean for all trials at each time for each neuron

    signalnoSqueeze = np.nanmax(signalnoSqueezenoMax , axis=0)
    #chooses the average signal, at the timepoint at which it's mean is maximum for each neuron

    signal = np.squeeze(signalnoSqueeze)
    SNR = signal/base
    trueMean = np.nanmean(matrix, axis=(0,1))
    trueSNR = trueMean/base


    if verbose:
        print(f"baseNoSqueeze: {baseNoSqueeze}")
        print(f"base: {base}")
        print(f"neuron 1: {np.nanstd(matrix[gt, :, 0])}")
        print(f"neuron 2: {np.nanstd(matrix[gt, :, 1])}")
        print(f"signalnoSqueezenoMax:\n{signalnoSqueezenoMax}")
        print(f"signalnoSqueeze: {signalnoSqueeze}")
        print(f"max mean signal: {signal}")
        print(f"SNR (signal/base): {SNR}")
        print(f"true signal: {trueMean}")
        print(f"true SNR (true signal/base): {trueSNR}")
        print("\n------------------------------------------\n")
    return base, signal

def testNormMatrix(matrix, signal, verbose=False):
    if verbose:
        print("\n---------------- testNormMatrix -----------------\n")
        print(f"matrix, with size: {matrix.shape}")
        printTLData(matrix)
        print(signal)
    
    # Normalize response and calculate data
    norm_matrix = []
    for i in range(matrix.shape[2]):
        norm_matrix.append(matrix[:, :, i] / signal[i])
    temp_norm_matrix = np.asarray(norm_matrix)
    #need to move axis, as currently temp_norm_matrix is shaped like (neurons, time, trials)
    norm_matrix_moved = np.moveaxis(np.asarray(norm_matrix),0,-1)
    #shape back to (time, trials, neurons) from the temporary state

    test = matrix.astype(np.float64)
    test /= signal.reshape(1, 1, len(signal)) #ensure signal is one dimensional, with length neurons
    #does the exact same thing as the above for loop

    if verbose:
        print(f"normalised matrix, with size: {temp_norm_matrix.shape}\nprinting in regular python print")
        print(temp_norm_matrix)
        print(f"test matrix, with size: {test.shape}\nprinting in TL format")
        printTLData(test)
        print(f"normalised and moved matrix, with size: {norm_matrix_moved.shape}\nprinting in TL format")
        printTLData(norm_matrix_moved)
        print("\n-------------------------------------------------\n")
    return test

def testValTrials(val_stims, inc, verbose=False):
    if verbose:
        print("\n---------------- testValTrials -----------------\n")
        print(f"val stims size: {val_stims.shape}\n {val_stims}")
    #val_stims goes in jumps of 10, only use is to convert it back to jumps of 1
    #this just serves to confuse as for all intents and purpose, val_trials is used later on
    #but val_stims is stored

    val_trials = []
    for i in range(len(val_stims)):
        val_trials.append(np.arange(val_stims[i], val_stims[i] + inc))
    val_trials = np.asarray(val_trials)
    val_idx = np.arange(val_trials.min(),val_trials.max())

    if verbose:
        print(f"val trials size: {val_trials.shape}\n {val_trials}")
        print(f"val idx size: {val_idx.shape}\n {val_idx}")

        print("\n------------------------------------------------\n")

    return val_trials, val_idx

def testAllTemps(val_trials, norm_resp, stims, gt,verbose = False):
    if verbose:
        print("\n---------------- testAllTemps -----------------\n")
        print(f"val trials size: {val_trials.shape}\n {val_trials}")
        print(f"stims size: {stims.shape}\n {stims}")
        print(f"gt size: {gt.shape}\n {gt}")
        print(f"norm resp size: {norm_resp.shape}")
        printTLData(norm_resp)
        i, j = 2, 3
        print(f"i, j: {i}, {j}")
        print(f"val trials[i, j]:   {val_trials[i, j]}")
        print(f"where stims = trials[i, j]:   {np.where(stims == val_trials[i, j])}")
        temp = norm_resp[gt, np.where(stims == val_trials[i, j]), :]
        print(f"norm resp, with above condition size:{temp.shape})")
        printTLData(temp)
        """important to note here:
            our initial array norm_resp has shape (time, trials, neurons)
            after sorting for the correct image trial via the stims value
            and gt sorting, we get a new array with shape: (trial, times, neurons)
                                                            (1, gt, neurons)
        """
        temp2 = norm_resp[gt, :, :]
        printTLData(temp2)
        
    ind = 0
    all_temps = []
    for i in range(val_trials.shape[0]):
        temp = []
        for j in range(val_trials.shape[1]):
            temp.append(np.squeeze(norm_resp[gt, np.where(stims == val_trials[ind, j]), :]))
            #see note above, this squeezing messes up formatting, i.e
        temp = np.asarray(temp)
        all_temps.append(np.squeeze(np.nanmean(temp, axis=1)))
        ind += 1
        
    all_temps_np = np.asarray(all_temps)
    all_reps = np.moveaxis(np.moveaxis(np.asarray(all_temps),0,-1),1,0)
    
    #the above can be written as:
    meanGT = np.nanmean(norm_resp[gt, :, :], axis=0)
    x,y = val_trials.shape
    _, _, nNeurons = norm_resp.shape
    meanGTCorrect = meanGT.reshape(x,y, nNeurons)
    stimsCorrect = stims.reshape(x,y)
    
    if verbose:

        print(f"all temps size: {len(all_temps)}\n {all_temps}")
        print(f"all temps as np size: {all_temps_np.shape}\n {all_temps_np}")
        #print(f"all reps size: {all_reps.shape}\n {all_reps}")

        print("\n-----------------------------------------------\n")

    return 

def testReliab(all_temps:np.ndarray, reps):
    reliab = []
    for i in range(all_temps.shape[-1]):
        temp = []
        for j in range(all_temps.shape[1]):
            temp.append(np.corrcoef(np.nanmean(np.squeeze(all_temps[:, reps != j, i]), axis=1),np.squeeze(all_temps[:, j, i]))[0, 1])
        reliab.append(np.asarray(temp))
    reliab = np.asarray(reliab)
    mean_reliab = np.mean(reliab, axis=1)
    
    return

def main():
    trials = 20
    neurons = 2
    tb = np.arange(0, 0.7, 0.15)
    length = len(tb)
    gt = np.logical_and(tb > 0.01, tb < 0.5)
    interval = 4
    reps = np.arange(0, interval)
    Realtrials = np.arange(0, trials)
    stims = np.arange(0, trials)
    np.random.shuffle(stims)

    matrix = np.arange((length*trials*neurons)).reshape(length,trials,neurons)
    zeros = np.zeros(length)
    matrix[:, 0, 0] = zeros
    
    base, signal = testSNR(matrix, gt, True)
    norm_matrix = testNormMatrix(matrix, signal)
    val_stims = np.arange(0, trials, interval)
    val_trials, val_idx = testValTrials(val_stims, interval)

    testAllTemps(val_trials, norm_matrix, stims, gt)
    #this is useless, why instantiate val_stims just to use it once and reshape back to og...
    #should just run val_trials = np.arange(3601, 4001) and be done with it
    #val_idx, i dont even know at this point why three arrays which are the same
    #are needed....
    exit()
    
    return

def weightMatrix():
    size = 7
    neurons = 2

    temp = np.arange((size*size*neurons), dtype=np.float64).reshape(neurons, size, size)
    tensor = torch.from_numpy(temp)
    tensor.requires_grad_(True)
    tensor = tensor.unsqueeze(0)
    print(tensor.shape)
    print(tensor)

    final_tensor = tensor.view(tensor.shape[0],tensor.shape[1],tensor.shape[2]*tensor.shape[3],1)
    print(final_tensor.shape)
    print(final_tensor)

    #detach the gradient, so its just a tensor of numbers
    #convert the tensor to a numpy array
    #squeeze out the excess dimensions of size 1
    xVals = final_tensor.detach().numpy().squeeze() 
    nNeurons, imgLen = xVals.shape
    print(xVals.shape)
    print(nNeurons, imgLen)
    print(xVals)

    reshapedXVals = xVals.reshape(nNeurons, size, size)
    meanXVals = np.mean(reshapedXVals, axis=0)
    print(meanXVals.shape)
    print(meanXVals)

    meanTemp = np.mean(temp, axis=0)
    """
    meanXVals = np.mean(xVals, axis=0)
    reshapedMeanXVals = meanXVals.reshape(size, size)
    print(reshapedMeanXVals.shape)
    print(reshapedMeanXVals)
    """
    for n in range(nNeurons):
        plt.imshow(reshapedXVals[n, :, :], cmap='bwr')
        plt.colorbar()        
        plt.show()
        pass
    
    plt.imshow(meanXVals, cmap='bwr')
    plt.colorbar() 
    plt.show()
    plt.imshow(meanTemp, cmap='bwr')
    plt.colorbar() 
    plt.show()
    return

if __name__ == '__main__':
    main()
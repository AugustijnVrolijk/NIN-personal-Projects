
dataFilename = ["Ajax_20241012_001_normcorr_SPSIG_Res.mat"]; %data to be trial locked
infoFilename = ["Ajax_20241012_001_normcorr_SPSIG.mat"]; %file with frame and line info
filepath = "C:\Users\augus\NIN Stuff\data\koenData\"; %path to files

fullFileData = fullfile(filepath, dataFilename);
fullFileInfo = fullfile(filepath, infoFilename);

data = load(fullFileData);
info = load(fullFileInfo);

rois = struct2table(info.info.rois);

outputVals = trialLockMult(info.info.frame, ... % array, 1* number of stimuli, frame at which stimuli happens
...
    data.sigCorrected, data.spike_prob,  data.spike_timesSig,...   %add however many signals you want triallocked
...
    lines=info.info.line, ... %array, 1* number of stimuli, line at which the frame is at
    py=rois.py(:), ... %array, 1* number of neurons, y position for neuron n
    framesBeforeStim=info.info.framesbeforstim, ... 
    framesAfterStim=info.info.framesafterstim);

[TLsig, TLspikeprob, TLspikeTime] = outputVals{:};    %outputVals, has all trial locked sigs in cell array in order you put them in

function outputSigs = trialLockMult(trials, inputSig, info)
    arguments (Input)
        trials (:, 1) double %input array showing frames at which trials occured 
    end
    arguments (Input, Repeating)
        inputSig (: , :) double %input signals 
    end
    arguments (Input)
        info.lines (:, 1) double = NaN %input array showing where the frame update line was at stimulus onset
        info.py (:, 1) double = NaN %input array showing the position of neuron x along the y dimension
        info.framesBeforeStim double = 8 %number of frames to show before stim
        info.framesAfterStim double = 16 %number of frames to show after stim
    end
    arguments (Output)
        outputSigs (:, :) cell
    end
    iter = length(inputSig);
    outputSigs = cell(iter, 1);
    for i=1:iter
        outputSigs{i} = trialLockSignal(inputSig{i}, trials, lines=info.lines, ...
        py=info.py(:),framesBeforeStim=info.framesBeforeStim, ...
        framesAfterStim=info.framesAfterStim);
    end
end

%{
xAxis = dataTL.Res.ax;

[meanSig, stdSig] = getTrialAvg(TLsig);
[spikeProbMeanSig, spikeProbStdSig] = getTrialAvg(TLspikeprob);
[spikeTimeMeanSig, spikeTimeStdSig] = getTrialAvg(TLspikeTime);


function plotTrials(trials, inputSig)
    arguments (Input)
        trials (:, 1) double %input array showing frames at which trials occured 
    end
    arguments (Input, Repeating)
        inputSig (: , :) double %input signals 
    end
    for i=1:1
        plot(xAxis, [spikeTimeMeanSig(:, 1), spikeProbMeanSig(:,1),meanSig(:,1)])
    end
end
%}


function [meanSig, stdSig] = getTrialAvg(TLsignal)
    arguments (Input)
        TLsignal (:, :, :)
    end
    arguments (Output)
        meanSig (:, :) double
        stdSig(:, :) double
    end
    trialMean = mean(TLsignal, 2);
    meanSig = squeeze(trialMean);
    trialDeviation = std(TLsignal, 0, 2);
    stdSig = squeeze(trialDeviation);
   
end

filename = ["Ajax_20241012_001_normcorr_SPSIG_Res.mat", "Ajax_20241012_001_normcorr_SPSIG.mat"];
filepath = "C:\Users\augus\NIN Stuff\data\koenData\";
fullFileRes = fullfile(filepath, filename(1));
fullFile = fullfile(filepath, filename(2));

dataTL = load(fullFileRes);
data = load(fullFile);

rois = struct2table(dataTL.info.rois);

outputVals = trialLockMult(dataTL.info.frame, ...
    data.sigCorrected, data.spike_prob,  data.spike_timesSig,...
    lines=dataTL.info.line, py=rois.py(:), ...
    framesBeforeStim=dataTL.info.framesbeforstim, ...
    framesAfterStim=dataTL.info.framesafterstim);

[TLsig, TLspikeprob, TLspikeTime] = outputVals{:};

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

function trialLockedSig = trialLockSignal(inputSig, trials, info)
     arguments (Input)
        inputSig (: , :) double %input signal 
        trials (:, 1) double %input array showing frames at which trials occured
        info.lines (:, 1) double = NaN %input array showing where the frame update line was at stimulus onset
        info.py (:, 1) double = NaN %input array showing the position of neuron x along the y dimension
        info.framesBeforeStim double = 8 %number of frames to show before stim
        info.framesAfterStim double = 16 %number of frames to show after stim
%}
xAxis = dataTL.Res.ax;

[meanSig, stdSig] = getTrialAvg(TLsig);
[spikeProbMeanSig, spikeProbStdSig] = getTrialAvg(TLspikeprob);
[spikeTimeMeanSig, spikeTimeStdSig] = getTrialAvg(TLspikeTime);

%{
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
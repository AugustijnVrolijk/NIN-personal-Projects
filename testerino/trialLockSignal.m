function trialLockedSig = trialLockSignal(inputSig, trials, info)
     arguments (Input)
        inputSig (: , :) double %input signal 
        trials (:, 1) double %input array showing frames at which trials occured
        info.lines (:, 1) double = NaN %input array showing where the frame update line was at stimulus onset
        info.py (:, 1) double = NaN %input array showing the position of neuron x along the y dimension
        info.framesBeforeStim double = 8 %number of frames to show before stim
        info.framesAfterStim double = 16 %number of frames to show after stim
    end
    arguments (Output)
        trialLockedSig (:, : , :) double
    end
    trimBeforeID = 1; %init to include in scope
    trimAfterID = length(trials); %init to include in scope
    
    [totalLen, nNeuron] = size(inputSig);

    checkTrialInput(trials, totalLen); %update trim vals to ensure validity
    
    trials = trials(trimBeforeID:trimAfterID);
    info.lines = info.lines(trimBeforeID:trimAfterID);
    
    nTrials = length(trials);
    
    %in format
    trialLockedSig = NaN((info.framesBeforeStim+info.framesAfterStim),nTrials,nNeuron);

    progress = waitbar(0, ['Get trials for ' num2str(nNeuron) ' rois and ' num2str(nTrials) ' stimuli.']);
    for t = 1:nTrials
        %we want the 8 frames before to be before the stimulus, as well as
        %including it and the 16 frames to be after the stimulus
        startIdx = trials(t) - info.framesBeforeStim + 1; %start id
        endIdx = trials(t) + info.framesAfterStim; %end id
        curLine = info.lines(t);

        %for each neuron check where the frame update line is. if neurons
        %are below it, they will update after they have actually seen the
        %stimulus, therefore shift them back by 1
        for n = 1:nNeuron
            if info.py(n) < curLine
                trialLockedSig(1:end, t, n) = inputSig((startIdx):(endIdx), n);
            else
                trialLockedSig(1:end, t, n) = inputSig((startIdx-1):(endIdx-1), n);
            end
        end
        waitbar(t/nTrials, progress);
    end
    close(progress);

    function checkTrialInput(trials, totalLen)
        if isnan(info.lines)
            warning("No value given for which line was being refreshed per frame \n" + ...
                "Trial locking will not account for frame refresh rate");
            info.lines = ones(length(trials));
            info.py = zeros(length(nNeuron));
        elseif length(trials) ~= length(info.lines)
            error("number of trials and number of frame line information is not equal");
        end
        
        if isnan(info.py)
            warning("No value given for Y position of each neuron \n" + ...
                "Trial locking will not account for frame refresh rate");
            info.py = zeros(length(nNeuron));
        elseif length(info.py) ~= nNeuron
            error("number of neurons in input signal and neuron positions are not equal");
        end

        while trials(trimAfterID) >= (totalLen - info.framesAfterStim)
            warning(['not enough poststim frames in trial:' num2str(trimAfterID)]);
            trimAfterID = trimAfterID - 1;
        end
        while trials(trimBeforeID) <= info.framesBeforeStim
            warning(['not enough poststim frames in trial:' num2str(trimBeforeID)]);
            trimBeforeID = trimBeforeID + 1;
        end
    end
end


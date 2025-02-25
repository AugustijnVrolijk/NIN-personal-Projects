function spk = SpikeVec_2_SpikeTimes(spkMat, frameTimes)
% spk = SpikeVec_2_SpikeTimes(spkMat, frameTimes)
% Convert spike times to a [nt x nrois] double where the number in each
% element denotes the number of spikes for that timepoint to spike times,
% where each element is a spike denoting the time each spike happened.
% 
% Input:
%   spkMat ([ntime x nrois] double). Spikecount for every time bin
% 
% Output:
%   spk ([nrois x 1] cell with [nSpikes x 1] double). Time indexes for
%                                                     every spike
%   
% To convert spike times to spike matrix see SpikeTimes_2_SpikeVec
% 
% 
% Leander de Kraker
% 2023-3-14
% 

nrois = size(spkMat, 2);

spk = cell(1, nrois);
for i = 1:nrois
    % Option 1
    takingrounds = 0;
    while any(spkMat(:, i)>0)
        spkTimes = find(spkMat(:,i)>0);
        spk{i} = cat(1, spk{i}, spkTimes);
        spkMat(:,i) = spkMat(:,i) - 1;
        takingrounds = takingrounds + 1;
    end
    if takingrounds>1
        spk{i} = sort(spk{i});
    end
    spk{i} = frameTimes(spk{i});

    % Option 2

end

if nrois == 1
    spk = spk{1};
end
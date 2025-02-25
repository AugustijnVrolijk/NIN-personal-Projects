function PlotTextSpikeCounter(spk, burstdelay, ytext, colortext)
% Plot number of spikes if neuron bursts/ spikes happen too close together
% on axis. Can also plot the spikes themselves depending on ytext input
% 
% Input:
%   spk ([nspikes x 1] double). Spike times as plotted on the plot axis
%   burstdelay (2 options):
%       ([2 x 1] double): the axis limit, to determine when to 
%       (scalar value): The time to 
%   ytext ([3 x 1] double). y value of text. y begin and end of spike line
%         (scalar double).  y value of text
%   
% 
% Effect:
%   Plots text in plot which says the amount of spikes, for if spikes are
%   happening in bursts
% 
% Leander de Kraker
% 2023-3-14
% 

if size(spk, 1)>size(spk, 2)
    spk = spk';
end

if length(burstdelay)>1
    burstdelay = abs(burstdelay(2)-burstdelay(1))./30;
end

nsp = length(spk);
if nsp>1
    delays = diff(spk);
    burst = [false, delays<burstdelay, false];
    burststart = find(diff(burst)>0);
    burstend = find(diff(burst)<0);
    burstCount = burstend-burststart+1;
    burstX = mean([spk(burststart); spk(burstend)]);

    nburst = length(burstX);
        
    for i=1:nburst
        text(burstX(i), ytext, num2str(burstCount(i)), ...
            'color', colortext, 'fontsize', 10, ...
            'verticalalignment', 'middle')
    end
end
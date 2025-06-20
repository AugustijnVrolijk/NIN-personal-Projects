 nRois = length(info.rois);
rfOnDist = zeros(nRois,1);
rfOffDist = zeros(nRois,1);
rfOnGlmIncl = zeros(nRois,1);
rfOffGlmIncl = zeros(nRois,1);
for n = 1:nRois
    fwhmOn = info.rois(n).onFWHM;
    fwhmOff = info.rois(n).offFWHM;
    azi = info.rois(n).azi;
    ele = info.rois(n).ele;
    rfsz = info.rois(n).rfsz;
    %brat = info.rois(n).BRAT;
    rsq = info.rois(n).RSQ;
    snr = info.rois(n).SNR;

    aziOnDist = -(azi(1)+fwhmOn/2); % because azi values are negative on left side of screen
    aziOffDist = -(azi(2)+fwhmOff/2); % because azi values are negative on left side of screen
    eleOnDist = ele(1)-fwhmOn/2;
    eleOffDist = ele(2)-fwhmOff/2;

    rfOnDist(n) = min(aziOnDist, eleOnDist);
    rfOffDist(n) = min(aziOffDist, eleOffDist);
    if rsq(1)>rsqThresh && snr(1)>snrThresh
        rfOnGlmIncl(n) = 1;
    end
    if rsq(2)>rsqThresh && snr(2)>snrThresh
        rfOffGlmIncl(n) = 1;
    end
end
% inclusion of neurons based on RF properties
onCrit = rfOnDist>rfDistVec & logical(rfOnGlmIncl);
offCrit = rfOffDist>rfDistVec & logical(rfOffGlmIncl);
rfInclPre = onCrit | offCrit; % either a good ON or OFF receptive field
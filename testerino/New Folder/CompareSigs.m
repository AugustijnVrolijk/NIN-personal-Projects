% Comparing signals like never before!!!
%
% Leander de Kraker
% 2023-2-2
%

clear

filepaths = {'C:\Users\augus\NIN Stuff\data\koenData\copy\';...
            };
filenames = {'Ajax_20241012_001_normcorr_SPSIG.mat';...
            };
sigNames = {{'sigCorrected', 'spike_prob', 'spike_timesSig', 'deconCorrected'}};
isSpiking = [false,             false,        true,            true,...
             ];

isSpikingSub = find(isSpiking);
nfiles = length(filenames);
nsigs = sum(cellfun(@length, sigNames));

sigAlias = cell(nsigs, 1);

counter = 1;
dataPre = cell(nfiles, 1);
for i = 1:nfiles
    dataPre{i} = load([filepaths{i}, filenames{i}], sigNames{i}{:});

    load([filepaths{i}, filenames{i}], 'frameTimes', 'freq', 'PP', 'BImgMax', 'BImg')
    for j = 1:length(sigNames{i})
        if contains(sigNames{i}{j}, 'seal')
            dataPre{i}.(sigNames{i}{j}) = dataPre{i}.(sigNames{i}{j}) .* 8;
        end
        sigAlias{counter} = [sigNames{i}{j}, sprintf('_%d', i)];
        data.(sigAlias{counter}) = dataPre{i}.(sigNames{i}{j});
        counter = counter + 1;
    end
end

if ~exist('PP', 'var')
    nrois = PP.Cnt;
else
    nrois = size(data.(sigAlias{1}), 2);
end
for i = 1:nsigs
    if size(data.(sigAlias{i}), 2) ~= PP.Cnt
        if size(data.(sigAlias{i}), 1) ~= PP.Cnt
            warning('different amount of ROIs for %s\n', sigAlias{i})
        else
            fprintf('Transposing %s\n', sigAlias{i})
            data.(sigAlias{i}) = data.(sigAlias{i})';
        end
    end
end
if exist('frameTimes', 'var')
    nt = length(frameTimes);
else
    frameTimes = 1:size(data.(sigAlias{1}), 1);
end
nt = length(frameTimes);

% Convert the 
for i = isSpikingSub
    if ~iscell(data.(sigAlias{i}))
        dataSpiking.(sigAlias{i}) = SpikeVec_2_SpikeTimes(data.(sigAlias{i}), frameTimes);
    else % We got the spiking data, need to convert to sig
        dataSpiking.(sigAlias{i}) = data.(sigAlias{i});
        if ~all(isinteger(data.(sigAlias{1})))
            mult = freq;
        else
            mult = 1;
        end
        data.(sigAlias{i}) = SpikeTimes_2_SpikeVec(round(dataSpiking.(sigAlias{i}).*mult), [nrois, nt]);
    end
end
clear dataPre i j



%% Rename for blablalbla

sigAliasOld = sigAlias;
sigAlias = {'sigCorrected', 'spike_prob', 'spike_times', 'MLspike'};

for i = 1:nsigs
    data.(sigAlias{i}) = data.(sigAliasOld{i});
    data = rmfield(data, sigAliasOld{i});
    if isSpiking(i)
        dataSpiking.(sigAlias{i}) = dataSpiking.(sigAliasOld{i});
        dataSpiking = rmfield(dataSpiking, sigAliasOld{i});
    end
end

data.sigCorrected = data.sigCorrected - 1;

for i = 1:nrois
    dataSpiking.spike_times{i} = dataSpiking.spike_times{i} + 1/freq;
end


%% Plot signals

tidx = 1:1000;
t = frameTimes(tidx)/freq;
% Activate much tighter subplots
% [subplot margin top&side],[figure bottomspace,topspace],[leftspace,rightspace]
subplot = @(m,n,p) subtightplot (m, n, p, [0.1 0.05], [0.175 0.1], [0.03 0.01]);

colors = lines(nsigs);
colors(4,:) = mean(colors(2:3,:));
yt = [0;...
      -.75;...
      -1.5];
yh = 0.3;
yht = 0.15;


lwidth = [2 1 1.5 1.5 1.5 1.5];

figure('Position', [5 217 1000 200],  'Renderer','painters')
% subplot(1,1,1)
subplot(1,4,4)
% imagesc(CreateRGB2({BImgMax, BImg}, [0 1 0; 0.5 0.5 1], true, true))
imagesc(BImg); colormap(gray)
hold on
for i = 1:nrois
    plot(PP.Con(i).x, PP.Con(i).y, 'r')
end    
hdot = plot(PP.P(1, 1), PP.P(1, 2), '.r');
set(gca, 'XTickLabel', [], 'YTickLabel', [])

for roi = 1:nrois
    subplot(1,4,[1 2 3])
    count = 0;
    for i = 1:nsigs
        if ~isSpiking(i)
            plot(t, data.(sigAlias{i})(tidx, roi), 'color', colors(i, :), 'LineWidth', lwidth(i))
            hold on
        else % For spiking signals plot lines
            count = count + 1;
            x = dataSpiking.(sigAlias{i}){roi}/freq;
            xt = x>t(1) & x<t(end);
            xt = x(xt);
            PlotTextSpikeCounter(xt, 0.2, yt(count)-yh-yht, colors(i,:))
            nSpikes = length(x);
            x = [x; x; NaN(1, nSpikes)];
            x = x(:);
            y = repmat([yt(count); yt(count)-yh; NaN], nSpikes, 1);
            plot(x, y, 'color', [colors(i, :) 0.5], 'linewidth', lwidth(i))
            
            str = sprintf('%d: %s', nSpikes, strrep(sigAlias{i}, '_', ' '));
            text(t(1), mean([yt(count) yt(count)-yh]), str, 'color', colors(i,:),...
                'FontSize', 11, 'FontWeight', 'bold')
        end
    end
    hold off
    title(sprintf('ROI %d', roi))
    legend(strrep(sigAlias(~isSpiking), '_', ' '))
%     xlim([0 30])
    xlim(t([1 end]))
    xlabel('Time (sec)')
    ylims = ylim;
    ylim([-2.5 max(ylims(2), 1.75)])
    figTitle = sprintf('%s_signals-ROI%d', filenames{1}, roi);

    subplot(1,4,4)
    delete(hdot)
    px = PP.P(1, roi);
    py = PP.P(2, roi);
    hdot = plot(px, py, '.r');
    xlim([px-40, px+40])
    ylim([py-40, py+40])
%     SaveImg('png', figTitle)
    pause
end

%% spike counts for every spiking signal
spikingNames = sigAlias(isSpikingSub);
nSpikes = zeros(nrois, length(isSpikingSub));
count = 0;
for i = 1:length(isSpikingSub)
    count = count + 1;
    for j = 1:nrois
        nSpikes(j, count) = length(dataSpiking.(spikingNames{i}){j});
    end
end

% 
figure('renderer', 'painters')
boxplot_w_paireddata(nSpikes, spikingNames, 8, 1:10:nrois, cmapL('italian roast', nrois));
ylabel('Number of spikes per ROI')

%%
clearvars i j px py roi str tidx yh yt x y ylims hdot lwidth sigAliasOld sigNames count counter figTitle t


%% Cross-correlations between signals

% remove 32 values of beginning and end of signals which could be NaNs for CASCADE signals
dataSmall = data;
for i = 1:nsigs
    dataSmall.(sigAlias{i}) = data.(sigAlias{i})(33:end-33, :);
end

count = 0;

laglim = 20;
isSpikingn = length(isSpikingSub);
xcorrs = cell(nsigs, nsigs);
tic
for i = 1:nsigs
    for j = i:nsigs
        xcorrs{i, j} = zeros(laglim*2+1, nrois);
        for roi = 1:nrois
            xcorrs{i,j}(:,roi) = xcorr(dataSmall.(sigAlias{i})(:,roi), dataSmall.(sigAlias{j})(:,roi), laglim);
        end
        count = count + 1;

        fprintf('Done with %d/%d, %d/%d. elapsed time %.1f minutes. ETA %.1f minutes\n',...
                    i, nsigs, j, nsigs, toc/60, ETA(count, sum(1:nsigs), toc/60))
    end
end
bla = zeros(nsigs);
count = 1;
for i = 1:nsigs
    for j = 1:i-1
%         xcorrs{i,j} = flipud(xcorrs{j,i});
        xcorrs{i, j} = zeros(laglim*2+1, nrois);

        bla(i,j) = count;
        count = count+1;
    end
end
%% Plot the signal cross correlations

% Activate much tighter subplots
% [subplot margin top&side],[figure bottomspace,topspace],[leftspace,rightspace]
subplot = @(m,n,p) subtightplot (m, n, p, [0.025 0.08], [0.1 0.1], [0.1 0.025]);


x = (-laglim:laglim)./freq;
h = gobjects(nsigs, nsigs);
counter = 1;
figure('Renderer','painters')
for i = 1:nsigs
    for j = 1:nsigs
        subplot(nsigs, nsigs, counter)
        if i==j
            colorLine = [1 0.4 0];
        else
            colorLine = [0 0.4470 0.7410];
        end
        h(i, j) = plot(x, xcorrs{i, j}(:,1), 'color', colorLine);
        hold on
        xline(0, 'k')
        % ylim([0 7000])
        counter = counter + 1;
        if j == 1
            ylabel(strrep(sigAlias{i}, '_', ' '),...
                'FontSize', 8.8, 'FontWeight', 'bold', 'Rotation',60)
        end
        if i == 1
            title(strrep(sigAlias{j}, '_', ' '))
        end
        if i == nsigs
            xlabel('Time (sec)')
        else
            set(gca, 'XTickLabel', [])
        end
        box off
    end    
    htitle = figtitle(sprintf('cross correlations ROI %d', 1));
end
pause

for roi = [8 9 13 15 16 28 38]
    counter = 1;
    delete(htitle)
    for i = 1:nsigs
        for j = 1:nsigs
            hold off

            h(i, j).YData = xcorrs{i, j}(:,roi);
            counter = counter + 1;
        end
    end
    str = sprintf('Amber_20220614_004_cross_correlations_ROI-%d', roi);
    htitle = figtitle(strrep(str, '_', ' '));
    SaveImg('png', str)
end

%% Plot the upper triangle signal cross correlations

% Activate much tighter subplots
% [subplot margin top&side],[figure bottomspace,topspace],[leftspace,rightspace]
subplot = @(m,n,p) subtightplot (m, n, p, [0.025 0.08], [0.1 0.1], [0.1 0.025]);

TODO

x = (-laglim:laglim)./freq;
h = gobjects(nsigs, nsigs);
counter = 0;
figure('Renderer','painters')
for i = toploti
    for j = toploti
        counter = counter + 1;
        subplot(nsigs, nsigs, counter)
        if i==j
            colorLine = [1 0.4 0];    
        else
            colorLine = [0 0.4470 0.7410];
        end
        h(i, j) = plot(x, xcorrs{i, j}(:,1), 'color', colorLine);
        hold on
        xline(0, 'k')
        ylim([0 7000])
        if j == 1
            ylabel(strrep(sigAlias{i}, '_', ' '),...
                'FontSize', 8.8, 'FontWeight', 'bold', 'Rotation',60)
        end
        if i == 1
            title(strrep(sigAlias{j}, '_', ' '))
        end
        if i == j
            xlabel('Time (sec)')
        else
            set(gca, 'XTickLabel', [])
        end
        box off
    end    
    htitle = figtitle(sprintf('cross correlations ROI %d', 1));
end
pause

for roi = [8 9 13 15 16 28 38]
    counter = 1;
    delete(htitle)
    for i = 1:nsigs
        for j = 1:nsigs
            hold off

            h(i, j).YData = xcorrs{i, j}(:,roi);
            counter = counter + 1;
        end
    end
    str = sprintf('Amber_20220614_004_cross_correlations_ROI-%d', roi);
    htitle = figtitle(strrep(str, '_', ' '));
    SaveImg('png', str)
end
%% Post spike time histogram from spiking signals to all signals


count = 0;

laglim = 20;
isSpikingn = length(isSpikingSub);
xcorrs = cell(nsigs, nsigs);
tic
for i = 1:nsigs
    for j = 1:nsigs
        xcorrs{i, j} = zeros(laglim*2+1, nrois);
        for roi = 1:nrois
            xcorrs{i,j}(:,roi) = xcorr(data.(sigAlias{i})(:,roi), data.(sigAlias{j})(:,roi), laglim);
        end
        count = count + 1;

        fprintf('Done with %d/%d, %d/%d. elapsed time %.1f minutes. ETA %.1f minutes',...
                    i, nsigs, j, nsigs, toc, ETA(count, nsigs*nsigs, toc/60))
    end
end


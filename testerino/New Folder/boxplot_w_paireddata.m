function [hBox, hScatter, hLines] = boxplot_w_paireddata(data, groups, dotSize, toDraw, colors)
% Boxplot but with individual datapoints
% [hBox, hScatter] = boxplot_w_paireddata(values, groups, dotSize, toDraw, colors)
% 
% Input:
%   data ([nEntries x groups] double): The data
%   groups ([group x 1] cell): label for each column of data (allowed to be empty)
%   dotSize (scalar value) or (nEntries x groups): plotting dot size
%   toDraw ([nToDraw x 1] double): which the paired data entries to draw
%                          connecting lines for
%   colors ([nEntries x 3] double): color for each (paired) data entry 
%                                   (allowed to be empty)
% 
% Output: A nice boxplot, plotted in the current axes
%   hBox, handle to the things of the boxplot
%   hScatter, handle to the data dots
%   hLines, handle to the lines
% 
% 
% See also, boxplot_w_data
% 
% Leander de Kraker
% 2021-9-21
% 

n = size(data, 1);
nGroups = size(data, 2);

% x coordinates for dots and lines
jitter = randn(n,1)/15;
jitter = repmat(jitter, [1, nGroups]);
x = repmat(1:nGroups, [n, 1]);
x = x + jitter;

if isempty(colors)
    colors = zeros(n, 3);
end

% scatterplot
hScatter = scatter(x(:), data(:), dotSize(:), repmat(colors, [nGroups, 1]), 'filled');
hold on


% Lineplot

nLines = length(toDraw);
hLines = gobjects(nLines, 1);
for i = 1:nLines
    hLines(i) = plot(x(toDraw(i),:), data(toDraw(i),:), 'color', colors(toDraw(i),:));
end


% For boxplot
y = data(:);
if isempty(groups)
    groups = 1:nGroups;
else
    if size(groups,1)>size(groups,2)
        groups = groups';
    end
    groups = repmat(groups, [n,1]);
end
groups = groups(:);

hBox = boxplot(y, groups, 'symbol', 'k');



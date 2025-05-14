clc;
close all;

% Define image processing parameters for each image
imageParams = {
    % filename, filter type, filter params, second filter (optional)
    {'p1.png', 'movmean', [10 5], []};
    {'p1_2.png', 'medfilt2', [22 1], []};
    {'p1_3.png', 'medfilt2', [20 1], []};
    {'p1_4.png', 'medfilt2', [16 1], []};
    {'p1_5.png', 'wiener2', [5 5], {'movmean', [12 1]}};
    {'p1_6.png', 'movmean', [10 1], {'movmean', [10 1]}};
    {'p1_7.png', 'movmean', [20 1], {'movmean', [10 1]}};
    {'p1_8.png', 'movmean', [13 1], {'movmean', [10 1]}};
};

fontSize = 12;

for i = 1:length(imageParams)
    params = imageParams{i};
    filename = params{1};
    filterType = params{2};
    filterParams = params{3};
    secondFilter = params{4};
    
    % Read and prepare image
    grayImage = imread(filename);
    if size(grayImage, 3) > 1
        grayImage = grayImage(:, :, 1); % Take green channel
    end
    
    % Create figure
    figure;
    subplot(2, 1, 1);
    imshow(grayImage, []);
    title(['Original Grayscale Image: ' filename], 'FontSize', fontSize);
    
    % Apply first filter
    if strcmp(filterType, 'wiener2')
        filteredImage = wiener2(grayImage, filterParams);
    else
        Recons = [];
        for g = 1:size(grayImage, 2)
            if strcmp(filterType, 'movmean')
                colFiltered = movmean(grayImage(:,g), filterParams);
            elseif strcmp(filterType, 'medfilt2')
                colFiltered = medfilt2(grayImage(:,g), filterParams);
            end
            Recons = [Recons, colFiltered];
        end
        filteredImage = Recons;
    end
    
    % Apply second filter if specified
    if ~isempty(secondFilter)
        secondFilterType = secondFilter{1};
        secondFilterParams = secondFilter{2};
        
        Recons = [];
        for g = 1:size(filteredImage, 2)
            if strcmp(secondFilterType, 'movmean')
                colFiltered = movmean(filteredImage(:,g), secondFilterParams);
            end
            Recons = [Recons, colFiltered];
        end
        filteredImage = Recons;
    end
    
    % Show filtered image
    subplot(2, 1, 2);
    imshow(filteredImage, []);
    title(['Filtered Grayscale Image: ' filename], 'FontSize', fontSize);
    
    % Special handling for p1_5.png
    if strcmp(filename, 'p1_5.png')
        set(gcf, 'Units', 'Normalized', 'OuterPosition', [0 0 1 1]);
        fontSize = 20;
    end
end
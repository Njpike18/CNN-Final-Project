%% Requires Deep Learning Toolbox Installed

%% --- Parameters ---
imageSize = [32 32 4]; % Height, Width, Channels (polarized light)
numClasses = 3;
shapeTypes = {'line', 'thin_surface', 'thick_surface'};

%% --- Define CNN Architecture ---
layers = [
    imageInputLayer(imageSize, 'Name', 'input', 'Normalization', 'zerocenter', 'Mean', 0)
    convolution2dLayer(3, 8, 'Padding','same', 'Name', 'conv1')
    reluLayer('Name', 'relu1')
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'pool1')
    fullyConnectedLayer(numClasses, 'Name', 'fc')
];

lgraph = layerGraph(layers);
net = dlnetwork(lgraph); % ✅ No output layer = VALID for dlnetwork

%% --- Visualize CNN Response ---
figure;
tiledlayout(length(shapeTypes), 2, 'TileSpacing', 'tight');

for i = 1:length(shapeTypes)
    % Generate input
    inputImage = createPolarizedShape(shapeTypes{i}, imageSize(1:2));
    dlInput = dlarray(single(inputImage), 'SSCB'); % HxWxCxB

    % Show input
    nexttile;
    imshow(mean(inputImage, 3), []);
    title(['Input: ' shapeTypes{i}]);

    % Get activation from first conv layer
    act = predict(net, dlInput, 'Outputs', 'conv1');

    % Format activations for montage
    numFilters = size(act, 3);
    filterImages = cell(1, numFilters);
    for j = 1:numFilters
        filterImages{j} = mat2gray(extractdata(act(:, :, j, 1)));
    end

    nexttile;
    montage(filterImages, 'Size', [ceil(sqrt(numFilters)) ceil(sqrt(numFilters))]);
    title(['Conv1 Activations: ' shapeTypes{i}]);
end

%% --- Function to simulate polarized input ---
function polarizedImage = createPolarizedShape(shapeType, imageSize)
    polarizedImage = zeros(imageSize(1), imageSize(2), 4); % 4 polarization channels
    baseIntensity = 0.5 * ones(imageSize(1), imageSize(2)); % background

    switch shapeType
        case 'line'
            center = round(imageSize(1)/2);
            baseIntensity(center-1:center+1, :) = 1;
            polarizedImage(:,:,1) = baseIntensity;
            polarizedImage(:,:,2) = baseIntensity * 0.8;
            polarizedImage(:,:,3) = baseIntensity * 0.6;
            polarizedImage(:,:,4) = baseIntensity * 0.7;

        case 'thin_surface'
            [X, Y] = meshgrid(1:imageSize(2), 1:imageSize(1));
            center = round(imageSize / 2);
            mask = (X - center(2)).^2 + (Y - center(1)).^2 <= 5^2;
            baseIntensity(mask) = 1;
            polarizedImage(:,:,1) = baseIntensity * 0.7;
            polarizedImage(:,:,2) = baseIntensity * 0.9;
            polarizedImage(:,:,3) = baseIntensity;
            polarizedImage(:,:,4) = baseIntensity * 0.8;

        case 'thick_surface'
            [X, Y] = meshgrid(1:imageSize(2), 1:imageSize(1));
            center = round(imageSize / 2);
            mask = (X - center(2)).^2 + (Y - center(1)).^2 <= 10^2;
            baseIntensity(mask) = 1;
            polarizedImage(:,:,1) = baseIntensity * 0.9;
            polarizedImage(:,:,2) = baseIntensity * 0.7;
            polarizedImage(:,:,3) = baseIntensity * 0.8;
            polarizedImage(:,:,4) = baseIntensity;

        otherwise
            error('Unknown shape type: %s', shapeType);
    end
end


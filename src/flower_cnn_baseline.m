%% Coursework: Flower Classification CNN (Baseline)
% Ashar Zeeshan

%% Step 0 — Dataset setup & cleanup
datasetPath = 'data/flowers'; % update this path to point to your local copy of the dataset

% Create datastore
imds = imageDatastore(datasetPath, ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');

% Count images per class
tbl = countEachLabel(imds);
disp(tbl);

% Preview random images
figure;
perm = randperm(length(imds.Files), 20);
for i = 1:20
    subplot(4,5,i);
    imshow(imds.Files{perm(i)});
end

% Check for broken/unreadable files
brokenFiles = {};
for i = 1:numel(imds.Files)
    filePath = imds.Files{i};
    try
        imfinfo(filePath);
    catch
        fprintf('Unreadable image: %s\n', filePath);
        brokenFiles{end+1} = filePath; %#ok<AGROW>
    end
end
fprintf('\nFound %d broken files.\n', numel(brokenFiles));

% (Optional) delete broken files
% for i = 1:numel(brokenFiles)
%     delete(brokenFiles{i});
% end

% Recreate clean datastore
imds = imageDatastore(datasetPath, ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');

%% Step 1 — Split dataset
[imdsTrain, imdsRest] = splitEachLabel(imds, 0.7, 'randomized');
[imdsValidation, imdsTest] = splitEachLabel(imdsRest, 0.5, 'randomized');

%% Step 2 — Define image size & augmentation
inputSize = [128 128 3];

imageAugmenter = imageDataAugmenter( ...
    'RandRotation', [-20 20], ...
    'RandXTranslation', [-3 3], ...
    'RandYTranslation', [-3 3], ...
    'RandXReflection', true);

augimdsTrain = augmentedImageDatastore(inputSize, imdsTrain, 'DataAugmentation', imageAugmenter);
augimdsValidation = augmentedImageDatastore(inputSize, imdsValidation);
augimdsTest = augmentedImageDatastore(inputSize, imdsTest);

%% Step 3 — Define CNN layers
layers = [
    imageInputLayer(inputSize)

    convolution2dLayer(3, 32, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2, 'Stride', 2)

    convolution2dLayer(3, 64, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2, 'Stride', 2)

    convolution2dLayer(3, 128, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2, 'Stride', 2)

    dropoutLayer(0.5)
    fullyConnectedLayer(numel(unique(imds.Labels))) % auto-match #classes
    softmaxLayer
    classificationLayer
];

%% Step 4 — Training options
options = trainingOptions('adam', ...
    'MaxEpochs', 20, ...
    'MiniBatchSize', 32, ...
    'InitialLearnRate', 1e-4, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', augimdsValidation, ...
    'ValidationFrequency', 30, ...
    'ValidationPatience', 5, ... % stop early if no improvement
    'Verbose', false, ...
    'Plots', 'training-progress');

%% Step 5 — Train network
net = trainNetwork(augimdsTrain, layers, options);

%% Step 6 — Evaluate on test set
[YPred, probs] = classify(net, augimdsTest);
accuracy = mean(YPred == imdsTest.Labels);
fprintf('Test accuracy: %.2f%%\n', accuracy*100);

% Confusion matrix
figure;
plotconfusion(imdsTest.Labels, YPred);
title('Confusion Matrix — Test Set');

%% Coursework: Flower Classification with ResNet-18 (Transfer Learning)
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

% Reload datastore
imds = imageDatastore(datasetPath, ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');

%% Step 1 — Split dataset
[imdsTrain, imdsRest] = splitEachLabel(imds, 0.7, 'randomized');
[imdsValidation, imdsTest] = splitEachLabel(imdsRest, 0.5, 'randomized');

%% Step 2 — Load pretrained ResNet-18
net = resnet18;
inputSize = net.Layers(1).InputSize;
numClasses = numel(categories(imds.Labels));

% Convert network to layer graph
lgraph = layerGraph(net);

% Remove last classification layers
lgraph = removeLayers(lgraph, {'fc1000','prob','ClassificationLayer_predictions'});

% Add new layers for flower classification
newLayers = [
    fullyConnectedLayer(numClasses, 'Name','fc', ...
        'WeightLearnRateFactor',10,'BiasLearnRateFactor',10)
    softmaxLayer('Name','softmax')
    classificationLayer('Name','classification')];

% Connect new layers
lgraph = addLayers(lgraph, newLayers);
lgraph = connectLayers(lgraph, 'pool5', 'fc');

%% Step 3 — Data augmentation & resizing
imageAugmenter = imageDataAugmenter( ...
    'RandRotation', [-20 20], ...
    'RandXReflection', true, ...
    'RandXTranslation', [-5 5], ...
    'RandYTranslation', [-5 5]);

augimdsTrain = augmentedImageDatastore(inputSize(1:2), imdsTrain, ...
    'DataAugmentation', imageAugmenter);
augimdsValidation = augmentedImageDatastore(inputSize(1:2), imdsValidation);
augimdsTest = augmentedImageDatastore(inputSize(1:2), imdsTest);

%% Step 4 — Training options
options = trainingOptions('adam', ...
    'MiniBatchSize', 32, ...
    'MaxEpochs', 10, ... % 10–15 usually enough for transfer learning
    'InitialLearnRate', 1e-4, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', augimdsValidation, ...
    'ValidationFrequency', 20, ...
    'ValidationPatience', 5, ...
    'Verbose', false, ...
    'Plots', 'training-progress');

%% Step 5 — Train the transfer learning model
trainedNet = trainNetwork(augimdsTrain, lgraph, options);

%% Step 6 — Evaluate on test set
[YPred, probs] = classify(trainedNet, augimdsTest);
accuracy = mean(YPred == imdsTest.Labels);
fprintf('ResNet-18 Test Accuracy: %.2f%%\n', accuracy*100);

% Confusion matrix
figure;
plotconfusion(imdsTest.Labels, YPred);
title('Confusion Matrix — Test Set (ResNet-18)');

% Show sample predictions
figure;
idx = randperm(numel(imdsTest.Files), 16);
for i = 1:16
    subplot(4,4,i);
    I = readimage(imdsTest, idx(i));
    imshow(I);
    label = YPred(idx(i));
    title(string(label));
end

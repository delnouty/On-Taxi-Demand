function [trainedClassifier, validationAccuracy] = ModelOneClassifier(trainingData)
inputTable = trainingData;
predictorNames = {'Region', 'AveDistance', 'AveCodeDistance', 'AveCodeFare', 'Intencity', 'IsInCity', 'Tempo', 'IsHoliday', 'TD'};
predictors = inputTable(:, predictorNames);
response = inputTable.Demand;
isCategoricalPredictor = [true, false, false, false, false, false, false, true, true];

% Train a classifier
% This code specifies all the classifier options and trains the classifier.
classificationTree = fitctree(...
    predictors, ...
    response, ...
    'SplitCriterion', 'deviance', ...
    'MaxNumSplits', 262, ...
    'Surrogate', 'off', ...
    'ClassNames', categorical({'Low'; 'Medium'; 'High'}, {'Low' 'Medium' 'High'}, 'Ordinal', true));

% Create the result struct with predict function
predictorExtractionFcn = @(t) t(:, predictorNames);
treePredictFcn = @(x) predict(classificationTree, x);
trainedClassifier.predictFcn = @(x) treePredictFcn(predictorExtractionFcn(x));

% Add additional fields to the result struct
trainedClassifier.RequiredVariables = {'AveCodeDistance', 'AveCodeFare', 'AveDistance', 'Intencity', 'IsHoliday', 'IsInCity', 'Region', 'TD', 'Tempo'};
trainedClassifier.ClassificationTree = classificationTree;
trainedClassifier.About = 'This struct is a trained model exported from Classification Learner R2020b.';
trainedClassifier.HowToPredict = sprintf('To make predictions on a new table, T, use: \n  yfit = c.predictFcn(T) \nreplacing ''c'' with the name of the variable that is this struct, e.g. ''trainedModel''. \n \nThe table, T, must contain the variables returned by: \n  c.RequiredVariables \nVariable formats (e.g. matrix/vector, datatype) must match the original training data. \nAdditional variables are ignored. \n \nFor more information, see <a href="matlab:helpview(fullfile(docroot, ''stats'', ''stats.map''), ''appclassification_exportmodeltoworkspace'')">How to predict using an exported model</a>.');

% Extract predictors and response
% This code processes the data into the right shape for training the
% model.
inputTable = trainingData;
predictorNames = {'Region', 'AveDistance', 'AveCodeDistance', 'AveCodeFare', 'Intencity', 'IsInCity', 'Tempo', 'IsHoliday', 'TD'};
predictors = inputTable(:, predictorNames);
response = inputTable.Demand;
isCategoricalPredictor = [true, false, false, false, false, false, false, true, true];

% Perform cross-validation
partitionedModel = crossval(trainedClassifier.ClassificationTree, 'KFold', 10);

% Compute validation predictions
[validationPredictions, validationScores] = kfoldPredict(partitionedModel);

% Compute validation accuracy
validationAccuracy = 1 - kfoldLoss(partitionedModel, 'LossFun', 'ClassifError');

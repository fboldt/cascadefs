function cova = classifier_ova(fsmethod, baseclassifier, numberofclassifiers)
%
% cova = classifier_ova(fsmethod, baseclassifier, numberofclassifiers)
%
% %Example:
% dataset=load('datasets/wine.m');
% baseclassifier = classifier_knn()
% fscriterion = fscriterion_wrapper(baseclassifier)
% fsmethod = feature_selection_ranking(fscriterion)
% numberofclassifiers = 3
% classifier = classifier_ova(fsmethod, baseclassifier, numberofclassifiers)
% validation = validation_holdout()
% [results confusion trtime tetime trainedclassifier] = ml_evaluate(dataset, classifier, validation)
%
% See: classifier_knn, validation_houldout, validation_crossvalidation, create_performance, ml_evaluate.
%
if exist('fsmethod')==1
  cova.fsmethod=fsmethod;
else
  cova.fsmethod=feature_selection_ranking();
end
if exist('baseclassifier')==1
  cova.baseclassifier=baseclassifier;
else
  if any(strcmp('classifier',fieldnames(cova.fsmethod.fscriterion)))
    cova.baseclassifier=cova.fsmethod.fscriterion.classifier;
  else
    cova.baseclassifier=classifier_knn();
  end
end
if exist('numberofclassifiers')==1
  cova.numberofclassifiers=numberofclassifiers;
else
  cova.numberofclassifiers=1;
end
cova.train=@covatrain;
cova.predict=@covapredict;
cova.constructor=@classifier_ova;
cova.classifiername='OVA - One versus All Ensemble Feature Selection';
cova.autotunning = false;
end

%%% Training function
function cova = covatrain(cova, dataset)
starttime=cputime;
fscla = classifier_fs(cova.fsmethod, cova.baseclassifier);
if cova.numberofclassifiers>1
randens = classifier_random_ensemble(fscla, cova.numberofclassifiers, 1, 0.9);
else
randens = classifier_random_ensemble(fscla, 1, 1);
end
ovacla = classifier_onevsall(randens); 
cova.trained_classifier = ovacla.train(ovacla, dataset);
count=1;
for j=1:length(cova.trained_classifier.trained_classifier)
  for i=1:length(cova.trained_classifier.trained_classifier{j})
    used_features{count} = cova.trained_classifier.trained_classifier{j}.trained_classifier{i}.selected_features;
    count=count+1;
  end
end
occurences = analysis_features(dataset, used_features);
features = [1:size(dataset,2)];
cova.selected_features = features(occurences>0);
cova.training_time=cputime-starttime;
end

%%% Prediction function
function [answers confidences] = covapredict(cova, dataset)
starttime=cputime;
[answers confidence] = cova.trained_classifier.predict(cova.trained_classifier,dataset);
time=cputime-starttime;
end


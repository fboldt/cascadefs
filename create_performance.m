function performance = create_performance(performance_name)
%
% performance = create_performance(performance_name)
%
% * Implemented performances:
% accuracy (acc)
% precision (ppv)
% recall or sensitivity (tpr)
% f1 score (f1)
% f1 score average (f1avg)
% specificity (tnr)
% gmean
%
% performance_name default: accuracy
% 
% %Example:
% dataset = load('datasets/iris.m');
% classifier = classifier_knn(1, 'euclidean');
% performance = create_performance('meangmean')
% validation = validation_holdout(0.5, randi(1000), true, performance);
% [results confusion] = ml_evaluate(dataset, classifier, validation)
% 
% See: classifier_knn, validation_houldout, ml_evaluate.
%
if exist('performance_name')==0
  %performance_name='f1avg';
  performance_name='meangmean';
end
performance.name=performance_name;
if numel(performance_name)==numel('acc') && all(lower(performance_name)=='acc') || numel(performance_name)==numel('accuracy') && all(lower(performance_name)=='accuracy')
  performance.execute=@performance_acc;
  return
end
if numel(performance_name)==numel('ppv') && all(lower(performance_name)=='ppv') || numel(performance_name)==numel('precision') && all(lower(performance_name)=='precision')
  performance.execute=@performance_ppv;
  return
end
if numel(performance_name)==numel('tpr') && all(lower(performance_name)=='tpr') || numel(performance_name)==numel('recall') && all(lower(performance_name)=='recall') || numel(performance_name)==numel('sensitivity') && all(lower(performance_name)=='sensitivity')
  performance.execute=@performance_tpr;
  return
end
if numel(performance_name)==numel('f1') && all(lower(performance_name)=='f1') || numel(performance_name)==numel('f1score') && all(lower(performance_name)=='f1score')
  performance.execute=@performance_f1score;
  return
end
if numel(performance_name)==numel('f1avg') && all(lower(performance_name)=='f1avg') || numel(performance_name)==numel('f1average') && all(lower(performance_name)=='f1average')
  performance.execute=@performance_f1average;
  return
end
if numel(performance_name)==numel('tnr') && all(lower(performance_name)=='tnr') || numel(performance_name)==numel('specificity') && all(lower(performance_name)=='specificity')
  performance.execute=@performance_tnr;
  return
end
if numel(performance_name)==numel('gmean') && all(lower(performance_name)=='gmean') 
  performance.execute=@performance_gmean;
  return
end
if numel(performance_name)==numel('gmeanavg') && all(lower(performance_name)=='gmeanavg') || numel(performance_name)==numel('meangmean') && all(lower(performance_name)=='meangmean')  
  performance.execute=@performance_meangmean;
  return
end
performance.execute=@performance_f1average;
end

function accuracy = performance_acc(confusion)
  accuracy = trace(confusion)/sum(sum(confusion));
end

function precision = performance_ppv(confusion) 
warning('off');
for c=1:length(confusion)
  precision(c) = confusion(c,c)/sum(confusion(c,:));
end
warning('on');
filter = isnan(precision);
precision(filter) = 1;  
end

function recall = performance_tpr(confusion) 
warning('off');
for c=1:length(confusion)
  recall(c) = confusion(c,c)/sum(confusion(:,c));
end
warning('on');
filter = isnan(recall);
recall(filter) = 1;   
end

function specificity = performance_tnr(confusion) 
warning('off');
for c=1:length(confusion)
  conditionNegative = sum(confusion(:))-sum(confusion(:,c));
  tn = sum(confusion(:))+confusion(c,c)-sum(confusion(:,c))-sum(confusion(c,:));
  specificity(c) = tn/conditionNegative;
end
warning('on');
end

function gmean = performance_gmean(confusion)
  sensitivity = performance_tpr(confusion);
  specificity = performance_tnr(confusion);
  gmean = (sensitivity.*specificity).^0.5;
end

function meangmean = performance_meangmean(confusion)
  meangmean = mean(performance_gmean(confusion));
end

function f1score = performance_f1score(confusion)
  precision = performance_ppv(confusion);
  recall = performance_tpr(confusion);
  warning('off') ;
  f1score = 2*((precision.*recall)./(precision+recall));
  warning('on');
end

function f1average = performance_f1average(confusion)
  precision = mean(performance_ppv(confusion));
  recall = mean(performance_tpr(confusion));
  f1average = 2*((precision.*recall)./(precision+recall));
end


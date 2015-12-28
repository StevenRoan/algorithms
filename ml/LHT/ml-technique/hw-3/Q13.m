
trainD = load('./data/hw3_train.dat');
testD = load('./data/hw3_test.dat');



function [input groundtruth] = reorgData(data)
  input = data(:, [1:1:columns(data)-1]);
  groundtruth = data(:, columns(data));
endfunction

function [giniV]  = getGiniIndexValue (subLabels, classes)
  giniV = 1;
  for i =1:1:2
    giniV -= (sum(subLabels==classes(i))/length(subLabels))^2;
  endfor
endfunction

function [impureWeight] = getBranchImpureWeight (subLabels, classes)
  if (length(subLabels) <= 0)
    impureWeight = 10000000000;%aribtrary large number
    return;
  endif
  [giniV] = getGiniIndexValue(subLabels, classes);
  impureWeight = length(subLabels) * giniV;
endfunction

function [subLabelA subLabelB] = splitData(inputFeat, stump, labels)
  subLabelA = labels(inputFeat < stump);
  subLabelB = labels(inputFeat > stump);
endfunction

function [criteria] = getCriteriaByStump (inputFeat, stump, labels, classes)
  [subLabelA subLabelB] = splitData (inputFeat, stump, labels);
  criteria = getBranchImpureWeight (subLabelA, classes) + getBranchImpureWeight(subLabelB, classes);
endfunction

function [minFeat minStump terminate] = getStumpWithMinCriteria(features, labels, classes)
  minStump = NaN;
  minFeat = NaN;
  minCriteria = 1000000;
  terminate = false;
  if (length(unique(labels))==1 || length(unique(features(:,1)))==1 ||length(unique(features(:,2)))==1)
    terminate = true;
    return;
  endif
  for i = 1:columns(features)
    feat = features(:, i);
    [sortedFeat idx]= sort(feat);
    mappedLabels = labels(idx);
    l = sortedFeat(1);
    for j =2:rows(sortedFeat)
      r = sortedFeat(j);
      tmpStump = (l+r)/2;
      [tmpCriteria] = getCriteriaByStump(sortedFeat, tmpStump, mappedLabels, classes);
      if (tmpCriteria < minCriteria)
        minStump = tmpStump;
        minCriteria = tmpCriteria;
        minFeat = i;
      endif
      l = sortedFeat(j);
    endfor
  endfor
endfunction


global treeForStumpArray = [];
global treeForFeatArray = [];
global treeForBaseHypothesisArray = [];
global branching = 0;
global leafDataSet = 0;

function [] = buildDecisionTree(inputFeat, labels, classes, myIdxInTree)
  global treeForStumpArray;
  global treeForFeatArray;
  global treeForBaseHypothesisArray;
  global branching;
  global leafDataSet;
  [minFeat minStump terminate] = getStumpWithMinCriteria(inputFeat, labels, classes);
  % printf('myIdxInTree:%d, terminate:%d\n', myIdxInTree, terminate);
  if (terminate)
    leafDataSet+=rows(labels);
    % printf('leaf: %d = %d\n', myIdxInTree, unique(labels));
    treeForBaseHypothesisArray(myIdxInTree) = unique(labels);
    % printf('leaf set size:%d\n',rows(labels));
    % pause;
    return;
  else
    % minStump
    branching ++;
  endif
  treeForStumpArray(myIdxInTree)  = minStump;
  treeForFeatArray(myIdxInTree) = minFeat;
  inputA= inputFeat(inputFeat(:, minFeat) < minStump,:);
  inputB= inputFeat(inputFeat(:, minFeat) >= minStump,:);
  labelA= labels(inputFeat(:, minFeat) < minStump);
  labelB= labels(inputFeat(:, minFeat) >= minStump);
  % printf('myIdxInTree:%d, fidx:%d, stump:%d\n', myIdxInTree, treeForFeatArray(myIdxInTree), treeForStumpArray(myIdxInTree));
  % pause;
  buildDecisionTree(inputA, labelA, classes, 2*myIdxInTree); % left child
  buildDecisionTree(inputB, labelB, classes, 2*myIdxInTree+1); % right child
endfunction

function [prediction] = getPrediction (features, nodeIdx)
  global treeForFeatArray;
  global treeForStumpArray;
  global treeForBaseHypothesisArray;
  prediction = 0;
  % printf('nodeIdx:%d, treeForFeatArray:%d \n', nodeIdx, treeForFeatArray(nodeIdx));
  feature = features(:, treeForFeatArray(nodeIdx));
  % features
  % printf('nodeIdx:%d, fidx:%d, f: %d, stump:%d\n', nodeIdx, treeForFeatArray(nodeIdx),feature, treeForStumpArray(nodeIdx));
  % disp(feature > treeForStumpArray(nodeIdx))
  % pause;
  if (feature > treeForStumpArray(nodeIdx))
    % disp('A')
    idx = (2*nodeIdx +1);
    % disp(idx)
    % pause;
    if ( idx > length(treeForFeatArray) || treeForFeatArray(idx)==0)
      % printf('nextIdx:%d, gt:%d\n', idx, treeForBaseHypothesisArray(idx));
      prediction = treeForBaseHypothesisArray(idx);
      return;
    endif
    [prediction] = getPrediction(features, idx);
  elseif (feature < treeForStumpArray(nodeIdx))
    % disp('B')
    idx = (2*nodeIdx);
    % pause;
    if (idx > length(treeForFeatArray) ||treeForFeatArray(idx)==0)
      prediction = treeForBaseHypothesisArray(idx);
      return;
    endif
    [prediction] = getPrediction(features, idx);
  endif
endfunction

function [errorRate] = getError(features, labels)
  errItem = 0;
  for i = 1:rows(features)
  % for i = 1:1
    feats = features(i, :);
    [prediction] = getPrediction(feats, 1);
    % printf('label: %d, prediction:%d\n', labels(i), prediction);
    if (labels(i) != prediction)
      errItem ++;
    endif
  endfor
  errorRate = errItem / rows(features);
endfunction

classes = [-1 1];
[input, labels] = reorgData(trainD);
buildDecisionTree(input, labels, classes, 1);
leafDataSet
branching
[errIn] = getError(input, labels)
[input, labels] = reorgData(testD);
[errOut] = getError(input, labels)
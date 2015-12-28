
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
  [myImpurity] = getBranchImpureWeight(labels, classes);
  if (myImpurity == 0)
    terminate = true
    return;
  endif
  for i = 1:columns(features)
    feat = features(:, i);
    [sortedFeat idx]= sort(feat);
    mappedLabels = labels(idx);
    l = sortedFeat(1) - 10;
    for j =1:rows(sortedFeat)
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
global branching = 0;


function [] = buildDecisionTree(inputFeat, labels, classes, myIdxInTree)
  global treeForStumpArray;
  global treeForFeatArray;
  global branching;
  if (length(labels) <= 1)
    return;
  endif
  [minFeat minStump terminate] = getStumpWithMinCriteria(inputFeat, labels, classes);
  if (terminate)
    return;
  else
    branching ++;
  endif
  treeForStumpArray(myIdxInTree)  = minStump
  treeForFeatArray(myIdxInTree) = minFeat
  inputA= inputFeat(inputFeat(:, minFeat) < minStump);
  inputB= inputFeat(inputFeat(:, minFeat) >= minStump);
  labelA= labels(inputFeat(:, minFeat) < minStump);
  labelB= labels(inputFeat(:, minFeat) >= minStump);
  buildDecisionTree(inputA, labelA, classes, 2*myIdxInTree); % left child
  buildDecisionTree(inputB, labelB, classes, 2*myIdxInTree+1); % right child
endfunction


classes = [-1 1];
[input, labels] = reorgData(trainD);
buildDecisionTree(input, labels, classes, 1);
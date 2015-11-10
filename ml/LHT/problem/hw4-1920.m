addpath('./lib');

trainData = load('./data/hw4-1320/train.dat');
N = rows(trainData);
X = trainData(:, [1:1:columns(trainData)-1]);
Y = trainData(:, columns(trainData));
X = [ones(rows(X),1) X];

lambdas =[];
for i =1:13
  lambdas(i) = 1000/10^i;
end

% Construct the left index and right(last) index in each interval
foldSize = 40;
iniIdx= [];
lastIdx = []
for i = 1: ceil(rows(X)/foldSize)
  stIdx = (i-1)*foldSize + 1;
  if (i*foldSize < rows(X))
    endIdx = i*foldSize;
  else
    endIdx = rows(X);
  end
  iniIdx(i) = stIdx;
  lastIdx(i) =endIdx;
end

lopt=0;
minErrVal=1;
for i= 1:columns(lambdas)
  lambda =lambdas(i);
  ErrCrossValidation = 0;
  for j= 1: columns(iniIdx)
    Xvalidate = X([iniIdx(j):1:lastIdx(j)], :);
    Yvalidate = Y([iniIdx(j):1:lastIdx(j)], :);
    if iniIdx(j) > 1
      Xtrain = X([1:1:iniIdx(j)-1], :);
      Ytrain = Y([1:1:iniIdx(j)-1], :);
    endif

    if (lastIdx(j) < rows(X) && iniIdx(j) <=1)
      Xtrain = X([lastIdx(j)+1:1:rows(X)], :);
      Ytrain = Y([lastIdx(j)+1:1:rows(X)], :);
    elseif (lastIdx(j) < rows(X))
      Xtrain = [Xtrain; X([lastIdx(j)+1:1:rows(X)], :)];
      Ytrain = [Ytrain; Y([lastIdx(j)+1:1:rows(Y)], :)];
    endif

    I = eye(columns(Xtrain));
    Wopt = inv(Xtrain'*Xtrain+lambda.*I)*Xtrain'*Ytrain;
    ErrRateTrain = getErrorForClassification(Xtrain, Wopt, Ytrain);
    ErrRateValidate= getErrorForClassification(Xvalidate, Wopt, Yvalidate);
    ErrCrossValidation += ErrRateValidate;
  end
  ErrCrossValidation = ErrCrossValidation / columns(iniIdx);
  if (ErrCrossValidation < minErrVal || (ErrCrossValidation == minErrVal && lambda > lopt))
    minErrVal = ErrCrossValidation;
    lopt = lambda;
  endif
end
% Q19 Ans, lopt, minCrossValidationError
Xtrain = X;
Ytrain = Y;
lambda = lopt
minCrossValidationError= minErrVal

I = eye(columns(Xtrain));
Wopt = inv(Xtrain'*Xtrain+lambda.*I)*Xtrain'*Ytrain;
testData = load('./data/hw4-1320/test.dat');
Xout = testData(:,[1:1:columns(testData)-1]);
Xout = [ones(rows(Xout),1) Xout];
Yout = testData(:, columns(testData));
%Q20 Ans
ErrRateIn = getErrorForClassification(X, Wopt, Y)
ErrRateOut = getErrorForClassification(Xout, Wopt, Yout)
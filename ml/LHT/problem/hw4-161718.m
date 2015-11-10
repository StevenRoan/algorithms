% HW4 Problem  16, 17, 18
addpath('./lib');

trainData = load('./data/hw4-1320/train.dat');
N = rows(trainData);
X = trainData(:, [1:1:columns(trainData)-1]);
Y = trainData(:, columns(trainData));
X = [ones(rows(X),1) X];

Xtrain = X([1:1:120] , :);
Ytrain = Y([1:1:120], :);
Xvalidate = X(121:1:rows(X), :);
Yvalidate = Y(121:1:rows(Y), :);

%16
lambdas =[];
for i =1:13
  lambdas(i) = 1000/10^i;
end

testData = load('./data/hw4-1320/test.dat');
Xout = testData(:,[1:1:columns(testData)-1]);
Xout = [ones(rows(Xout),1) Xout];
Yout = testData(:, columns(testData));

lopt=0;
minErrVal=1;
for i= 1:columns(lambdas)
  lambda =lambdas(i);
  I = eye(columns(Xtrain));
  Wopt = inv(Xtrain'*Xtrain+lambda.*I)*Xtrain'*Ytrain;
  ErrRateTrain = getErrorForClassification(Xtrain, Wopt, Ytrain);
  ErrRateValidate= getErrorForClassification(Xvalidate, Wopt, Yvalidate);
  if (ErrRateValidate < minErrVal)
    minErrVal = ErrRateValidate;
    lopt = lambda;
  endif
  ErrRateOut = getErrorForClassification(Xout, Wopt, Yout);
end
lopt
I = eye(columns(X));
Wopt = inv(X'*X+lopt.*I)*X'*Y;
ErrRateIn = getErrorForClassification(X, Wopt, Y)
ErrRateOut = getErrorForClassification(Xout, Wopt, Yout)
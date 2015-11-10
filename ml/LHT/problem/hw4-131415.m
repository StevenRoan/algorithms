addpath('./lib');

trainData = load('./data/hw4-1320/train.dat');
N = rows(trainData);
X = trainData(:, [1:1:columns(trainData)-1]);
Y = trainData(:, columns(trainData));
X = [ones(rows(X),1) X];

% Visualize the data
% K = X(Y==1, [1 2]);
% K2 = X(Y==-1, [1 2]);
% plotData(K(:,1),K(:, 2), 'rx');
% hold on;
% plotData(K2(:,1),K2(:, 2), 'bo');

% Realize the importance of Constan term
lambda =10;
I = eye(columns(X));
Wopt = inv(X'*X+lambda.*I)*X'*Y;
ErrRateIn = getErrorForClassification(X, Wopt, Y)

testData = load('./data/hw4-1320/test.dat');
Xout = testData(:,[1:1:columns(testData)-1]);
Xout = [ones(rows(Xout),1) Xout];
Yout = testData(:, columns(testData));
ErrRateOut = getErrorForClassification(Xout, Wopt, Yout)

%14
lambdas =[];
for i =1:13
  lambdas(i) = 1000/10^i;
end

for i= 1:columns(lambdas)
  lambda =lambdas(i)
  Wopt = inv(X'*X+lambda.*I)*X'*Y;
  ErrRateIn = getErrorForClassification(X, Wopt, Y)

  testData = load('./data/hw4-1320/test.dat');
  Xout = testData(:,[1:1:columns(testData)-1]);
  Xout = [ones(rows(Xout),1) Xout];
  Yout = testData(:, columns(testData));
  ErrRateOut = getErrorForClassification(Xout, Wopt, Yout)
end
trainD = load('./data/adaboost_train.dat');
testD = load('./data/adaboost_test.dat');

function [input groundtruth] = reorgData(data)
  input = data(:, [1:1:columns(data)-1]);
  groundtruth = data(:, columns(data));
endfunction

% get weighted error init u = [1/N, 1/N,...1/N]
function e = getWeightedErrorOfDecisionStump(s, i, theta, input, groundtruth, u)
  answer = ones(rows(input),1);
  answer(input(:,i) < theta)= -1;
  answer = s.* answer;
  errorItem = zeros(rows(input), 1);
  errorItem(answer!=groundtruth) = 1;
  errorItem = errorItem .* u;
  e = sum(errorItem)/sum(u); % NOTE normalized error rate
endfunction

function u = getNextU (s, i, theta, input, groundtruth, lastu, e)
  answer = ones(rows(input),1);
  answer(input(:,i) < theta)= -1;
  answer = s.* answer;
  r = ((1-e)/e)^0.5;
  u = zeros(rows(input),1);
  u(answer!=groundtruth) = lastu(answer!=groundtruth) * r;
  u(answer==groundtruth) = lastu(answer==groundtruth) / r;
endfunction

% do the prediction by g
function prediction = predictYbyg(colVecS, colVecI, colVecTheta, colVecInput)
  prediction = colVecS .* sign(colVecInput(colVecI).-colVecTheta);
endfunction

% do the prediction by G
function G = predictYbyG(colVecS, colVecI, colVecTheta, colVecAlpha, singleInput)
  colVecPrediction = predictYbyg(colVecS, colVecI, colVecTheta, singleInput');
  G = sign(sum(colVecAlpha.*colVecPrediction));
endfunction

function e = getErrorRateByG(colVecS, colVecI, colVecTheta, colVecAlpha, input, groundtruth)
  errCount = 0;
  for j = 1: rows(input)
    predictionOfG = predictYbyG(colVecS, colVecI, colVecTheta, colVecAlpha, input(j, :));
    if (groundtruth(j) != predictionOfG)
      errCount ++;
    endif
  endfor
  e = errCount / rows(input);
endfunction

% for adaboost decision stump, first round with lastError = 1/2, (e/1-e)^1/2  = 1
function [ finalS finalI finalTheta, minError] = findOptimalParameterByData(data, u)
  S = [1, -1];
  minError = NaN;
  [input groundtruth] = reorgData(data);
  for i = 1: columns(input)
    feature = input(:, i);
    [f idx] = sort(feature);
    l = f(1)-10;%10 can be arbitarary random number
    for j =1: rows(f)
      r = f(j);
      theta = (l+r)/2;
      for k = 1: columns(S)
        s = S(k);
        e = getWeightedErrorOfDecisionStump(s, i, theta, input, groundtruth, u);
        if (isnan(minError) || e < minError)
          minError = e;
          % printf('j:%d, s:%d, i:%d, theta:%f (l:%f, r:%f) er:%f\n', j, s, i, theta,l,r, e);
          finalS = s;
          finalI = i;
          finalTheta = theta;
        endif
      endfor
      l = f(j);
    endfor
  endfor
endfunction

%Q13
[input groundtruth] = reorgData (trainD);

u = ones(rows(input),1).* (1/rows(input));
T = 300;
colVecS = zeros(T,1);
colVecI = ones(T,1);
colVecTheta = zeros(T,1);
colVecAlpha = zeros(T,1);
[ inputTest groundtruthTest] = reorgData(testD);
minEin = 1;
minEout =1;
for t=1:T;
   t
   U= sum(u)
   [s, i, theta, e] = findOptimalParameterByData(trainD, u);
   e = e
   if (minEin>e)
    minEin = e
   endif
   % s= s
   % i=i
   % theta=theta
   colVecS(t) = s;
   colVecI(t) = i;
   colVecTheta(t) = theta;
   r = ((1-e)/e)^0.5;
   colVecAlpha(t) = 0.5*log((1-e)/e);
   u = getNextU (s, i, theta, input, groundtruth, u, e);
   E = getErrorRateByG(colVecS, colVecI, colVecTheta, colVecAlpha, input, groundtruth)
   Eout = getErrorRateByG(colVecS, colVecI, colVecTheta, colVecAlpha, inputTest, groundtruthTest)
   if (minEout>Eout)
    minEout = Eout;
   endif
endfor
minEin = minEin

minEout = minEout




% e = getErrorOfDecisionStump(i, s, theta, input, groundtruth);
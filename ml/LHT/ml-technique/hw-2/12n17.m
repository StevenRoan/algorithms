trainD = load('./data/adaboost_train.dat');
testD = load('./data/adaboost_test.dat');

function [input groundtruth] = reorgData(data)
  input = data(:, [1:1:columns(data)-1]);
  groundtruth = data(:, columns(data));
endfunction

function e = getErrorOfDecisionStump(s, i, theta, input, groundtruth)
  answer = ones(rows(input),1);
  answer(input(:,i) < theta)= -1;
  answer = s.* answer;
  errorItem = zeros(rows(input), 1);
  errorItem(answer!=groundtruth) = 1;
  e = sum(errorItem) / rows(groundtruth);
endfunction

function [ finalS finalI finalTheta, minError] = findOptimalParameterByData(data)
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
        e = getErrorOfDecisionStump(s, i, theta, input, groundtruth);
        if (isnan(minError) || e < minError)
          minError = e;
          printf('j:%d, s:%d, i:%d, theta:%f (l:%f, r:%f) er:%f\n', j, s, i, theta,l,r, e);
          finalS = s;
          finalI = i;
          finalTheta = theta;
        endif
      endfor
      l = f(j);
    endfor
  endfor
endfunction
%Q12
[input groundtruth] = reorgData(trainD);
[s, i, theta, minError] = findOptimalParameterByData(trainD)





[input groundtruth] = reorgData(testD);
e = getErrorOfDecisionStump( s, i, theta, input, groundtruth)
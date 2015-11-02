% Training data set
D = load('./data/hw3-1820/train.dat');
% Testing data set
TD = load('./data/hw3-1820/test.dat');

function o = logisticFunc (s)
o = 1 ./ (1.+exp(-1.*s));
endfunction

function ER = errRate (TD, W)
  %Testing
  TX = TD(:,[1:1:columns(TD)-1]);
  TY = TD(:, columns(TD));
  THY = TX * W;
  THY(THY>0) =1;
  THY(THY<=0) = -1;
  faultVec = zeros(columns(TX), 1);
  faultVec(THY!=TY) = 1;
  ER = sum(faultVec)/rows(TX);
endfunction

X = D(:,[1:1:columns(D)-1]);
Y = D(:, columns(D));
W = zeros(columns(X), 1);
% training
for i = 1:2000
  if i > 1000
    i = i -1000;
  endif
  gradient = zeros(columns(X), 1);
  k = logisticFunc(-1.*Y(i).*X(i,:)*W).*-Y(i).*X(i,:);
  gradient = gradient + k';
  W = W - 0.001.*gradient;
  disp(i)
ER = errRate(TD, W)
endfor
ER = errRate(TD, W)


% Training data set
D = load('./data/hw3-1820/train.dat');
% Testing data set
TD = load('./data/hw3-1820/test.dat');

function o = logisticFunc (s)
o = 1 / (1+exp(-1*s));
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
  gradient = zeros(columns(X), 1);
  for j=1:rows(X)
    k = logisticFunc(-1.*Y(j).*X(j,:)*W).*-Y(j).*X(j,:);
    gradient = gradient + k';
  endfor
   gradient = gradient./rows(X);
  W = W - 0.001.*gradient;
  W
  i
  ER = errRate(TD, W)
endfor
ER = errRate(TD, W)


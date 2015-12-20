% sample = 2, w_0 ~ 0.25
% sample = large number w_0 = 1


T = 1000;
d = 1;
sample = 1000;
allTheta = zeros(d+1,1);
for t = 1:T
  t
  X = unifrnd(0, 1, sample, d);
  Y = X.*X;
  X = [X ones(rows(X), 1)];
  theta = (pinv(X'*X))*X'* Y;
  allTheta += theta;
endfor
and = allTheta./T
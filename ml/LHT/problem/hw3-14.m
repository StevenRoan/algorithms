function y = targetFunction(x1, x2)
 radiusSq = x1.^2 + x2.^2;
 len = length(x1);
 idx = radiusSq > 0.6;
 y = ones(len, 1);
 y = -1.*y;
 y(idx) = 1;
 noise = unifrnd(0,1, len,1);
 idx = noise < 0.1;
 y(idx) = y(idx) .* -1;
endfunction

% G is the learned function. Here is learned from linear regression
function y = hypothesisG(theta, X);
 len = length(X);
 y = zeros(len, 1);
 for i=1:columns(X)
  y = y + theta(i).* X(:, i);
 endfor

 y(y>=0) = 1;
 y(y<0) = -1;
endfunction
% transform to qudratic function -> (1, x_1, x_2, x_1x_2, x_1^2, x_2^2)
function X = qudraticTransform (x1, x2)
len = length(x1);
X = [ones(len,1) x1 x2 x1.*x2 x1.^2 x2.^2];
endfunction

%%%%%%%%%Training%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
numOfData = 1000;
K = unifrnd(-1,1,numOfData, 2);
Y = targetFunction(K(:,1), K(:,2));
X = qudraticTransform(K(:,1), K(:,2));
theta = (pinv(X'*X))*X'*Y;

%%%%%%%%%%%%%%%Testing%%%%%%%%%%%%%%%
Kout = unifrnd(-1,1,numOfData, 2);
Yout = targetFunction(Kout(:,1), Kout(:,2));
Xout = qudraticTransform(Kout(:,1), Kout(:,2));

yLinearRegression = hypothesisG(theta, Xout);
fault = sum(yLinearRegression!=Yout);
errRate = fault / numOfData

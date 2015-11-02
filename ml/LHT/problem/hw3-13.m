
%Target function: the function which we want to learn
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
function y = hypothesisG(theta, x1, x2)
 len = length(x1);
 y = theta(1).* ones(len,1) + theta(2).*x1 + theta(3).*x2;
 y(y>=0) = 1;
 y(y<0) = -1;
endfunction

function plotData(x,y)
plot(x,y,'rx','MarkerSize',8); % Plot the data
end

erRate = 0;
for i = 1:1000
numOfData = 1000;
K = unifrnd(-1,1,numOfData, 2);
Y = targetFunction(K(:,1), K(:,2));<„„></„„>
X = [ones(numOfData,1) K];
theta = (pinv(X'*X))*X'*Y;
GY = hypothesisG(theta, K(:,1), K(:,2));
faultVector = GY != Y;
fault = sum(faultVector);
erRate += fault/numOfData;
disp(i)
endfor
erRate = erRate/1000


function y = errorFunc(u, v)
 y = exp(u) + exp(2*v) + exp(u*v) + u^2  -2*u*v + 2*v^2 - 3*u -2*v;
endfunction

function  y =  firstDerivative(u, v)
  y = [exp(u)+v*exp(u*v)+2*u-2*v-3; 2*exp(2*v)+u*exp(u*v)-2*u+4*v-2];
endfunction

function  y =  secondDerivative(u, v)
  y = [exp(u)+v^2*exp(u*v)+2 exp(u*v)+u*v*exp(u*v)-2; exp(u*v) + u*v*exp(u*v)-2 4*exp(2*v)+u^2*exp(u*v)+4];
endfunction
% Q7
% w = zeros(2,1);
% err = errorFunc(w(1), w(2))
% for i=1:5
%   w = w-0.001.*firstDerivative(w(1), w(2))
%   err = errorFunc(w(1), w(2))
% endfor

%Q10
w = zeros(2,1);
for i=1:5
y = secondDerivative(w(1), w(2));
w = w - inv(y)*firstDerivative(w(1),w(2))
err = errorFunc(w(1), w(2))
endfor
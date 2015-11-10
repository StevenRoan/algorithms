function errRate = getErrorForClassification(X, W, Y)
  Yest = X*W;
  Yest(Yest>0) = 1;
  Yest(Yest<0) = -1;
  N = rows(X);
  Er = sum(Yest != Y);
  errRate = Er/N;
endfunction
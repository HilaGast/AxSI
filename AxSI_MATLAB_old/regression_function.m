function [err]=regression_function(x, ydata,pixpredictH, pixpredictR, pixpredictCSF, prcsf)
xt=1-x(1)-prcsf;
newdata=x(2).*(x(1)*pixpredictH+ xt*pixpredictR+prcsf*pixpredictCSF);
err=(newdata-ydata);
err=err(:);

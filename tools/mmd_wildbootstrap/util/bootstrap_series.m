function [ processes ] = bootstrap_series(length,numPaths)
%BOOTSTRAP_SERIES generates the wild bootstrap process. For the details
%  see 'A Wild Bootstrap for Degenerate Kernel Tests' http://arxiv.org/abs/1408.5404
%Kacper Chwialkowski 

ln=20;
ar = exp(-1/ln);
variance = 1-exp(-2/ln);
model = arima('Constant',0,'AR',{ar},'Variance',variance);
processes = simulate(model,length,'numPaths',numPaths);

end


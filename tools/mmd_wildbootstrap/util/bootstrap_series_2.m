function [ processes ] = bootstrap_series_2(length,numPaths)
%BOOTSTRAP_SERIES generates the wild bootstrap process (for MATLAB <8.4). For the details
%  see 'A Wild Bootstrap for Degenerate Kernel Tests' http://arxiv.org/abs/1408.5404
%Kacper Chwialkowski 

ln=20;
ar = exp(-1/ln);
variance = 1-exp(-2/ln);

w=sqrt(variance)*randn(length,numPaths);
a = [1 -ar];
processes=filter(1,a,w);

end
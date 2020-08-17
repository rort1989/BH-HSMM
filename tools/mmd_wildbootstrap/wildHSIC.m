% Syntax
% ---
% ```
% RESULTS = wildHSIC(X,Y)
% RESULTS = wildHSIC(X,Y,Name,Value)
% ```
% **Arguments**
% 
% X and Y are arrays of the observations. Each row contains a single observation, each column contains single dimension of all observations. X and Y must have the same number of observations.
% 
% The named arguments can be any of:
% 
% + **'Test'** : if 1 the first flavor of the test is used, if 2 the second flavor is used - the differences between flavors are described in 'A Wild Bootstrap for Degenerate Kernel Tests'. The default is value is 1.   
% * **'Alpha'** : test level, default value is 0.05.
% * **'Kernel'**: function that takes two arrays X,Y, both of dimension m by d, and returns kernel matrix K of size m by m such that K(i,j) = k(X(i,:),Y(j,:)) where k is a kernel function. See rbf_dot for the default implementation.
% * **'WildBootstrap'**: function that returns wild bootstrap process. See bootstrap_series function in util directory for the default implementation.
% * **'NumBootstrap'** : number of bootstrap re-sampling. Default value is 300.
%   
% **Output**
% The Output RESULTS contains the following fields:
% * **testStat** : the value of test statistic.
% * **quantile** : test critical value.
% * **reject**  : 1 iff null hypothesis is rejected by the test.     
function [results] = wildHSIC(X,Y,varargin)
addpath('util')

okargs =   {'TestType','Alpha', 'Kernel_X','Kernel_Y' ,'WildBootstrap','NumBootstrap'};
defaults = {true,0.05, rbf_dot(X),rbf_dot(Y),@bootstrap_series_2, 300};
    [test_type,alpha, kernel_X,kernel_Y,wild_bootstrap, numBootstrap] = ...
        internal.stats.parseArgs(okargs, defaults, varargin{:});  

m=size(X,1);
assert(m == size(Y,1))
assert(test_type==1 || test_type==2)

K = kernel_X(X,X);
L = kernel_Y(Y,Y);
H = eye(m)-1/m*ones(m,m);
Kc = H*K*H;
Lc = H*L*H;
statMatrix = Kc.*Lc;

results = bootstrap_null(m,numBootstrap,statMatrix,alpha,wild_bootstrap,test_type);

end




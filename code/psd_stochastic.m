function [x_new, alpha] = psd_stochastic(x_old, grad, alpha, direct)
% function perform projected steepest update
% a stochastic constraint is enforced by the variable x, i.e. sum_i x_i=1
% reference: Robert M. Freund, 2004, Projection Methods for Linear Equality 
% Constrained Problems
% direct indicates whether we are doing minimization or maximization
% problem: must be 1 (maximization) or -1 (minimization)

% step 1
x_old = x_old(:);
assert(abs(sum(x_old)-1)<=1e-14,'input must be a stochastic vector')
O = length(x_old);

% step 2: solve direction-finding problem (DFP)
AA = [eye(O) ones(O,1)];
last = [ones(1,O) 0];
AA = [AA; last]; % (O+1)x(O+1)
bb = [grad(:); 0];
dd = AA\bb;
d = dd(1:O);
% assert(abs(sum(d))<=1e-14,'projected gradient does must be orthogonal to stochastic constraint')
% if abs(sum(d))>1e-14
%     display('projected gradient must be orthogonal to stochastic constraint');
% end

% step 3: line search for step size (not implement, simply set to small value,
% suit for SGD based optimization) and ensure updated value between 0 and 1
% ratio test
gap = x_old; % value to boundary, either 0 or 1 depending on d, which determine direction of change
if direct > 0
    gap(d>0) = 1-gap(d>0); % if d is positive, the gap is between x and 1, otherwise between x and 0
else
    gap(d<0) = 1-gap(d<0);
end
ratio = gap ./ abs(d);
% alpha = min(alpha(:),ratio); % Ox1 vector
alpha = min(alpha(:),min(ratio)); % scalar

% step 4: update variable
x_new = x_old + direct*alpha.*d;
x_new(x_new<0) = 0; %~ ensure non-negative
x_new = x_new/sum(x_new);
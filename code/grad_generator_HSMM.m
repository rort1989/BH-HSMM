function [GRAD, flag_valid] = grad_generator_HSMM(params, datacells, weight)
% compute gradient of generator HDM
% difference compared to grad_generator_HDM is the transition
% hyperparameters are reparameterized

% find valid index of data
flag_valid = ~isnan(weight) & ~isinf(weight) & (weight ~= 0);
[O,Q] = size(params.mu);
N = sum(flag_valid);
idx = find(flag_valid > 0);
GRAD_pi = zeros(Q,1);
GRAD_A = zeros(Q,Q);
GRAD_kappa_C = zeros(Q,1);
GRAD_mu = zeros(O,Q);
GRAD_kappa = zeros(O,Q);
count = N;
% theta = size(datacells{1},2)/Q;

%%%%%%%%%%%%%%%%% initial state, covariance
for j = idx(:)' % iterate all the instances in dataset
    % step G1-1: compute gradient per sequence
    [grad_pi, grad_A, grad_kappa_C, grad_mu, grad_kappa, flag] = gradient_HSMM(datacells{j}, params.prior, params.transmat, params.duramat, params.mu, params.sigma, params.mixmat);
    if flag || sum(isnan(grad_pi))>0 || sum(sum(isnan(grad_A)))>0 || sum(isnan(grad_kappa_C))>0 || sum(isinf(grad_pi))>0 || sum(sum(isinf(grad_A)))>0 || sum(isinf(grad_kappa_C))>0
        count = count - 1;
        fprintf('WARNING: Gradient of HSMM on sample %d is corrupted, skip.\n',j);
        % possible causes is the sampled parameters from generator is not good
        %%%%%%%% even worse case: count = 0, all the data are way off
        continue;
    end
    % gradient of prior
    % prior: d(log-Dirichlet)/d(prior), use uniform prior this becomes 0
    % transmat: d(log-Dirichlet)/d(transmat), use uniform prior this becomes 0
    % duration: d(log-Gamma)/d(transmat), use a broad prior, k=1, theta=T/2
%     GRAD_kappa_C = -params.duramat/hyperparams.theta
    % emission mean: d(log-Gaussian)/d(mu), l2 norm
%     GRAD_mu = -hyperparams.inv_sigma_0.*(params.mu-hyperparams.mu0); % OxQ .* OxQ
    % emission std: not modeled
    
    % noise term
%     noise_prior
%     noise_transmap
%     noise_kappa_C
%     noise_mu = 2*alpha*randn(O,Q); % alpha is the momentum constant
%     noise_kappa
    
    % step G1-2: accumulate gradients across all sequences
    GRAD_pi = GRAD_pi + grad_pi*weight(j);
    GRAD_A = GRAD_A + grad_A*weight(j);
    GRAD_kappa_C = GRAD_kappa_C + grad_kappa_C*weight(j);
    GRAD_mu = GRAD_mu + grad_mu*weight(j);
    GRAD_kappa = GRAD_kappa + grad_kappa*weight(j);
    
    
end

% store gradient
GRAD = cell(5,1);
if count > 0
    GRAD{1} = GRAD_pi/count;
    GRAD{2} = GRAD_A/count; % notice this is row-wise gradient
    GRAD{3} = GRAD_kappa_C/count;
    GRAD{4} = GRAD_mu/count;
    GRAD{5} = GRAD_kappa/count;    
else
    GRAD{1} = zeros(Q,1);
    GRAD{2} = zeros(Q,Q); % notice this is row-wise gradient
    GRAD{3} = zeros(Q,1);
    GRAD{4} = zeros(O,Q);
    GRAD{5} = zeros(O,Q);    
end

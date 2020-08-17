function [params,V] = updateParams_HSMM_SGHMC(params,GRAD,RHO,V,alpha)
% SGHMC update for an individual HSMMs using momentum
% v <- (1-alpha)v + learning_rate * new_grad
% params <- params
% V is the gradient of previous iteration
% GRAD is the new gradient of current iteration
% RHO is the learning rate
% use projected gradient for initial state and transition parameters

if nargin < 5
    alpha = 0.1;
end

%%%%%% the following scheme uses momentum to update GRAD
Q = length(params.prior);
% prior
rho = RHO(1);
V1 = (1-alpha)./rho.*V{1}+GRAD{1};
[params.prior,rho] = psd_stochastic(params.prior, V1, rho, -1); %.dyn%GRAD.pi
V{1} = rho.*V1;
% duramat
V{3} = (1-alpha)*V{3} + RHO(3).*GRAD{3}; 
params.kappa_C = params.kappa_C - V{3}; %.mu% GRAD.mu
% emission mean
V{4} = (1-alpha)*V{4} + RHO(4).*GRAD{4};
params.mu = params.mu - V{4}; %.mu% GRAD.mu
% emission std
V{5} = (1-alpha)*V{5} + RHO(5).*GRAD{5};
params.kappa = params.kappa - V{5}; %.kappa% GRAD.kappa
for q = 1:Q
    rho = RHO(2);
    V2 = (1-alpha)./rho.*V{2}(q,[1:q-1 q+1:Q]) + GRAD{2}(q,[1:q-1 q+1:Q]);
    [update,rho] = psd_stochastic(params.transmat(q,[1:q-1 q+1:Q]), V2, rho, -1); %.dyn% GRAD.A(q,:)
    if sum(update) == 0 || sum(isnan(update))>0
        display('error,skip update');
    else
        V{2}(q,[1:q-1 q+1:Q]) = rho'.*V2;
        params.transmat(q,[1:q-1 q+1:Q]) = update'; %%%%%% notice this is row-wise gradient
    end
    % reparameterization to get covariance
    params.sigma(:,:,q) = diag(exp(2*params.kappa(:,q)));
end
% reparameterization to get duramat
params.duramat = exp(params.kappa_C);

end
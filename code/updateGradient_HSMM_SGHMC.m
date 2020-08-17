function GRAD = updateGradient_HSMM_SGHMC(GRAD,params,hyperparams,alpha,batch_size)
% running average of GRAD
% add prior and noise term to current HSMM model

%     gradient of prior
%     prior: d(log-Dirichlet)/d(prior), use uniform prior this becomes 0
%     transmat: d(log-Dirichlet)/d(transmat), use uniform prior this becomes 0
%     duration: d(log-Gamma)/d(transmat), use a broad prior, k=1, theta=T/2
    GRAD{3} = GRAD{3} - params.duramat/hyperparams.theta/batch_size;
%     emission mean: d(log-Gaussian)/d(mu), l2 norm
    GRAD{4} = GRAD{4} - hyperparams.inv_sigma_0.*(params.mu-hyperparams.mu0)/batch_size; % OxQ .* OxQ
%     emission std: not modeled
    [O,Q] = size(params.mu);
    % noise term
    % noise_prior
    prob = dirichletrnd(ones(Q,1));
    GRAD{1} = GRAD{1} + 2*alpha*prob/batch_size;%;%
    % noise_transmap
    for q = 1:Q
        prob = dirichletrnd(ones(1,Q-1));
        GRAD{2}(q,[1:q-1 q+1:Q]) = GRAD{2}(q,[1:q-1 q+1:Q]) + 2*alpha*prob/batch_size;%;
    end
    % noise_kappa_C
    GRAD{3} = GRAD{3} + 2*alpha*0.1*randn(Q,1)/batch_size;%;
    % noise_mu
    GRAD{4} = GRAD{4} + 2*alpha*randn(O,Q)/batch_size; %;% alpha*0.01 is the momentum constant
    % noise_kappa
    
end
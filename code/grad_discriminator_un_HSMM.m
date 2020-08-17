function GRAD = grad_discriminator_un_HSMM(params_set,dataset_r,dataset_g,en_r,en_g,prob_r,prob_g)
% compute gradient of discriminator
% assume Number of samples of real data and fake data is the same
% modified from 'grad_discriminator_un_new'

% positive model gradient
[O,Q] = size(params_set(1).params.mu);
[N,K] = size(prob_r);
GRAD = cell(K,1);
% theta = size(dataset_r{1},2)/Q;

for k = 1:K % each HSMM
    params = params_set(k).params;
    count_r = 0;
    count_g = 0;
    % initial
    GRAD_pi_r = zeros(Q,1); 
    GRAD_pi_g = GRAD_pi_r;
    GRAD{k}{1} = GRAD_pi_r;
    % transition
    GRAD_A_r = zeros(Q,Q); 
    GRAD_A_g = GRAD_A_r;
    GRAD{k}{2} = GRAD_A_r;
    % duration
    GRAD_kappa_C_r = zeros(Q,1); 
    GRAD_kappa_C_g = GRAD_kappa_C_r;
    GRAD{k}{3} = GRAD_kappa_C_r;
    % emission mean
    GRAD_mu_r = zeros(O,Q); 
    GRAD_mu_g = GRAD_mu_r;
    GRAD{k}{4} = GRAD_mu_r;
    % emission std
    GRAD_kappa_r = zeros(O,Q); 
    GRAD_kappa_g = GRAD_kappa_r;
    GRAD{k}{5} = GRAD_kappa_r;
    
    c1 = en_r.*prob_r(:,k) - log(prob_r(:,k)+eps).*prob_r(:,k); % Nx1
    idx = find(isnan(c1)==0);
    for i = idx'
        % step 1: compute gradient per sequence
        [grad_pi_r, grad_A_r, grad_kappa_C_r, grad_mu_r, grad_kappa_r, flag_r] = gradient_HSMM(dataset_r{i}, params.prior, params.transmat, params.duramat, params.mu, params.sigma, params.mixmat);
        % step 2: accumulate gradients across all sequences
        if ~flag_r && sum(isnan(grad_pi_r))==0 && sum(sum(isnan(grad_A_r)))==0 && sum(isnan(grad_kappa_C_r))==0 && sum(isinf(grad_pi_r))==0 && sum(sum(isinf(grad_A_r)))==0 && sum(isinf(grad_kappa_C_r))==0
            count_r = count_r + 1;
            GRAD_pi_r = GRAD_pi_r + c1(i)*grad_pi_r;
            GRAD_A_r = GRAD_A_r + c1(i)*grad_A_r;
            GRAD_kappa_C_r = GRAD_kappa_C_r + c1(i)*grad_kappa_C_r;
            GRAD_mu_r = GRAD_mu_r + c1(i)*grad_mu_r;
            GRAD_kappa_r = GRAD_kappa_r + c1(i)*grad_kappa_r;
        end        
    end    
    
    c2 = en_g.*prob_g(:,k) - log(prob_g(:,k)+eps).*prob_g(:,k); % Nx1
    idx = find(isnan(c2)==0);
    for j = idx'
        % step 1: compute gradient per sequence
        [grad_pi_g, grad_A_g, grad_kappa_C_g, grad_mu_g, grad_kappa_g, flag_g] = gradient_HSMM(dataset_g{j}, params.prior, params.transmat, params.duramat, params.mu, params.sigma, params.mixmat);
        % step 2: accumulate gradients across all sequences
        if ~flag_g && sum(isnan(grad_pi_g))==0 && sum(sum(isnan(grad_A_g)))==0 && sum(isnan(grad_kappa_C_g))==0 && sum(isinf(grad_pi_g))==0 && sum(sum(isinf(grad_A_g)))==0 && sum(isinf(grad_kappa_C_g))==0
            count_g = count_g + 1;
            GRAD_pi_g = GRAD_pi_g + c2(j)*grad_pi_g;
            GRAD_A_g = GRAD_A_g + c2(j)*grad_A_g;
            GRAD_kappa_C_g = GRAD_kappa_C_g + c2(j)*grad_kappa_C_g;
            GRAD_mu_g = GRAD_mu_g + c2(j)*grad_mu_g;
            GRAD_kappa_g = GRAD_kappa_g + c2(j)*grad_kappa_g;
        end
    end
    if count_r > 0
        GRAD{k}{1} = -GRAD_pi_r/count_r;
        GRAD{k}{2} = -GRAD_A_r/count_r; % notice this is row-wise gradient
        GRAD{k}{3} = -GRAD_kappa_C_r/count_r;
        GRAD{k}{4} = -GRAD_mu_r/count_r;
        GRAD{k}{5} = -GRAD_kappa_r/count_r;
    end
    if count_g > 0
        GRAD{k}{1} = GRAD{k}{1} + GRAD_pi_g/count_g;
        GRAD{k}{2} = GRAD{k}{2} + GRAD_A_g/count_g; % notice this is row-wise gradient
        GRAD{k}{3} = GRAD{k}{3} + GRAD_kappa_C_g/count_g;
        GRAD{k}{4} = GRAD{k}{4} + GRAD_mu_g/count_g;
        GRAD{k}{5} = GRAD{k}{5} + GRAD_kappa_g/count_g;
    end
    
    % gradient of prior
    % prior: d(log-Dirichlet)/d(prior), use uniform prior this becomes 0
    % transmat: d(log-Dirichlet)/d(transmat), use uniform prior this becomes 0
    % duration: d(log-Gamma)/d(transmat), use a broad prior, k=1, theta=T/2
%     GRAD{k}{3} = GRAD{k}{3} - params.duramat/hyperparams.theta
    % emission mean: d(log-Gaussian)/d(mu), l2 norm
%     GRAD{k}{4} = GRAD{k}{4} - -hyperparams.inv_sigma_0.*(params.mu-hyperparams.mu0); % OxQ .* OxQ
    % emission std: not modeled
    
    % noise term
%     noise_mu = 2*alpha*randn(O,Q); % alpha is the momentum constant
end


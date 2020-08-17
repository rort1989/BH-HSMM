function [en_r,en_g,prob_r,prob_g] = coefficient_discriminator_HSMM_un(dataset_train_r,dataset_train_g,params,L,dura_type)
% comput coefficient used to update discriminator with unknown data label
% params_p is a set of parameters, each correspond to one class 

N = length(dataset_train_r);
M = length(dataset_train_g);
K = length(params);
llh_r = zeros(N,K);
llh_g = zeros(M,K);
for k = 1:K
    llh_r(:,k) = compute_llh_evidence_HSMM(dataset_train_r, params(k).params, L, 'dura_type', dura_type, 'scale', 1);%~ no mask_missing 
    llh_g(:,k) = compute_llh_evidence_HSMM(dataset_train_g, params(k).params, L, 'dura_type', dura_type, 'scale', 1);%~
end
% compute the conditional distribution of class label
s_r = logsumexp(llh_r,2);
s_g = logsumexp(llh_g,2);
prob_r = exp(llh_r-repmat(s_r,1,K)); % probability of positive data % NxK
prob_g = exp(llh_g-repmat(s_g,1,K)); % probability of negative data % NxK
%%%%%% verify each row of prob_r and prob_g should sum to 1

% compute entropy
en_r = sum(prob_r.*log(prob_r+eps),2); % Nx1
en_g = sum(prob_g.*log(prob_g+eps),2); % Nx1
en_r(isnan(en_r)) = 0;
en_r(en_r>0) = 0;
en_g(isnan(en_g)) = 0;
en_g(en_g>0) = 0;
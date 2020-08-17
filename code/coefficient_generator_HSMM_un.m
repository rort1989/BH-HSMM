function [g,prob_r,loglikelihood] = coefficient_generator_HSMM_un(datasets,params_dis,L,dura_type)
% compute the coefficient for generator HDM

K = length(params_dis);
loglikelihood = zeros(length(datasets),K);
for k = 1:K
    loglikelihood(:,k) = compute_llh_evidence_HSMM(datasets, params_dis(k).params, L, 'dura_type', dura_type, 'scale', 1); % num_sample_g * 1 vector
end
s_r = logsumexp(loglikelihood,2);
%%%%%%%%%%%%%%%%%%%%% verify this
% offset_r = max(loglikelihood_p,s_r);
% prob_r = exp(loglikelihood_p-offset_r) ./ exp(s_r-offset_r); % probability of datasets being positive
prob_r = exp(loglikelihood-repmat(s_r,1,K)); % probability of datasets being positive
% equivalently, use log(prob_g)

% manipulation in log domain to avoid underflow
g = sum(prob_r.*log(prob_r+eps),2); % num_sample_g * 1 vector
g(isnan(g)) = 0;
g(g>0) = 0;

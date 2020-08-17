function [obs,hid] = sample_HSMM(params,num_sample,T,scheme)
% function draw samples from HSMM using modified ancestral sampling
% First draw hidden state chain following Tokuda et al. 2013, then draw
% data from parameterized model
%%%%%% optionally: improve synthesized data using 'estimate_Position'

O = size(params.mu,1);
obs = cell(num_sample,1);
% obs = zeros(O,T,num_sample);
hid = zeros(num_sample,T);

ini_state = sample_discrete(params.prior,1,num_sample);
for i = 1:num_sample    
    hid(i,:) = sample_path(ini_state(i),params.transmat,params.duramat,T,scheme);
    obs{i} = zeros(O,T);
    for t = 1:T
        q = hid(i,t);
        m = sample_discrete(params.mixmat(q,:), 1, 1);
        obs{i}(:,t) =  gaussian_sample(params.mu(:,q,m), params.sigma(:,:,q,m), 1);
    end
end

% datasets.observed = obs;
% datasets.hidden = hid;
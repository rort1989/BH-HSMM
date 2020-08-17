function [datasets, params_g] = sample_hdm(params,num_params,num_sample,T,viterbicode)
% function draw samples from hierarchical dynamic model (HDM) using
% ancestral sampling i.e. first draw parameters from its prior, then draw
% data from parameterized model
% Strictly speaking, params include both parameters and hyperparameters 

if nargin < 5
    viterbicode = [];
end
[O,Q] = size(params.mu0);
obs = zeros(O,T,num_params*num_sample);
hid = zeros(num_params*num_sample,T);
%prior = zeros(Q,num_params*num_sample);
transmat = zeros(Q,Q,num_params*num_sample);
mu = zeros(O,Q,num_params*num_sample);
params_g = repmat(struct('prior',[],'sigma',[],'mixmat',[],'transmat',[],'mu',[]),num_params,1);

for i = 1:num_params % num of set of params
    params_g(i).prior = params.prior;
    params_g(i).sigma = params.sigma;
    params_g(i).mixmat = params.mixmat;
    % draw parameters from hyperparameters (except initial state dist. and emission cov dist.)    
    for q = 1:Q
        params_g(i).transmat(q,:) = dirichletrnd(params.eta(q,:));
        params_g(i).mu(:,q) = mvnrnd(params.mu0(:,q),params.sigma0(:,:,q));        
        % params_g.mixmat % mixture is not considered yet
    end
    % draw data given current parameters
    if isempty(viterbicode)
        dataset_ = synthetic_HMM(1,num_sample,T,Q,O,'params',params_g(i));
    else
        dataset_.hidden = repmat(viterbicode(:)',num_sample,1);
        for j=1:num_sample
            for t=1:T
                q = dataset_.hidden(j,t);
                m = sample_discrete(params_g(i).mixmat(q,:), 1, 1);
                dataset_.observed(:,t,j) =  gaussian_sample(params_g(i).mu(:,q,m), params_g(i).sigma(:,:,q,m), 1);
            end
        end
    end
    idx = (i-1)*num_sample+1:i*num_sample;
    % deposit parameters
    %prior(:,idx) = repmat(params_g.prior,1,num_sample);
    transmat(:,:,idx) = repmat(params_g(i).transmat,[1,1,num_sample]);
    mu(:,:,idx) = repmat(params_g(i).mu,[1,1,num_sample]);
    % deposit data
    obs(:,:,(i-1)*num_sample+1:i*num_sample) = dataset_.observed;
    hid((i-1)*num_sample+1:i*num_sample,:) = dataset_.hidden;
end

datasets.observed = obs;
datasets.hidden = hid;
%datasets.prior = prior;
datasets.transmat = transmat;
datasets.mu = mu;
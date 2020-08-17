function params_ini = combine_params(params,Qs,O,M,dura_type)
% combine initialization among different actions into one 
K = length(params); % number of classes
Q_ = sum(Qs);
mu = zeros(O,Q_*M);
sigma = zeros(O,O,Q_*M);
kappa = zeros(O,Q_*M);
mixmat = zeros(Q_,M);
prior = zeros(Q_,1);
transmat = zeros(Q_,Q_);
if strcmp(dura_type,'Multinomial')
    L = size(params(1).params.duramat,2);
    duramat = zeros(Q_,L);
elseif strcmp(dura_type,'Poisson')
    duramat = zeros(Q_,1);
    kappa_C = zeros(Q_,1);
else
    error('Unsupported duration distribution.')
end
q = 0;
qm = 0;
for k = 1:K
    prior(q+1:q+Qs(k)) = params(k).params.prior;
    transmat(q+1:q+Qs(k),q+1:q+Qs(k)) = params(k).params.transmat;
    duramat(q+1:q+Qs(k),:) = params(k).params.duramat;
    if strcmp(dura_type,'Poisson')
        kappa_C(q+1:q+Qs(k),:) = params(k).params.kappa_C;
    end
    mu(:,qm+1:qm+Qs(k)*M) = params(k).params.mu;
    sigma(:,:,qm+1:qm+Qs(k)*M) = params(k).params.sigma;
    kappa(:,qm+1:qm+Qs(k)*M) = params(k).params.kappa;
    mixmat(qm+1:qm+Qs(k)*M,:) = params(k).params.mixmat;
    q = q+Qs(k);
    qm = qm+Qs(k)*M;
end

params_ini.prior = prior/sum(prior);
params_ini.transmat = transmat;
params_ini.duramat = duramat;
if strcmp(dura_type,'Poisson')
    params_ini.kappa_C = kappa_C;
end
params_ini.mu = mu;
params_ini.sigma = sigma;
params_ini.kappa = kappa;
params_ini.mixmat = mixmat;
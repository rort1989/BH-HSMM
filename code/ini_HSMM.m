function params_ini = ini_HSMM(datacells, hstate_num, feature_dim, mixture, max_dura, varargin)
% function to perform initial estimation of parameters given data
% currently only support continuous observations
% initialize by clustering within each individual sequence and match the
% correspondence among different sequences

% optional input: mixture, discrete
% format dataset
% assume in format of cells

% load optional input and format data
p = inputParser;
default_replicate = 5;
default_cov_type = 'diag';
default_cov_prior = 0;
default_dura_prior = 0;
default_dura_type = 'Multinomial';
default_mask_missing = [];
addOptional(p,'replicate',default_replicate,@isnumeric);
addOptional(p,'cov_type',default_cov_type,@ischar);
addOptional(p,'cov_prior',default_cov_prior,@isnumeric);
addOptional(p,'dura_prior',default_dura_prior,@isnumeric);
addOptional(p,'dura_type',default_dura_type,@ischar);
addOptional(p,'mask_missing',default_mask_missing,@iscell);
p.parse(varargin{:});
replicate = p.Results.replicate;
cov_type = p.Results.cov_type;
cov_prior = p.Results.cov_prior;
dura_prior = p.Results.dura_prior;
dura_type = p.Results.dura_type;
mask_missing = p.Results.mask_missing;
if length(mask_missing) ~= length(datacells) % in this case, use complete data
    mask_missing = cell(length(datacells),1);
end

% Define RV dimenstion
N = length(datacells);
O = feature_dim;
Q = hstate_num;
L = max_dura;
M = mixture;

% initialization of parameters
first_slice_idx = zeros(N+1,1);
crt_first_slice_idx = 1;
data_all = zeros(O,N*1000);
dist = zeros(Q,Q,N);
IDX_all = zeros(10000,1); % pre-allocate size for speed, later truncated to the actual length

for n = 1:N
    % identify and only use the complete frames
    if ~isempty(mask_missing{n})
        % complete the sequence using mean value
        Vm = sum(datacells{n}.*mask_missing{n},2)./sum(mask_missing{n},2); % TxO  %%%%% assert Vm = mean(data_train{n},2) for mask_train{n} = ones(O,T)
        datacells{n} = datacells{n} + Vm(:,ones(1,size(datacells{n},2))).*(1-mask_missing{n});
    end
    first_slice_idx(n) = crt_first_slice_idx;
    crt_first_slice_idx = crt_first_slice_idx + size(datacells{n},2);
    data_all(:,first_slice_idx(n):crt_first_slice_idx-1) = datacells{n};  % D*total_num_slices
    
    % sequence-level clustering results
    [IDX, C] = kmeans(datacells{n}',Q,'Replicates',replicate);   
    
    % compute distance
    if n == 1 % can change this number i.e. set another sequence as reference
        mu_ref = C; % OxQ       
        for p = 1:Q
            data = datacells{n}(:,IDX==p);
            for q = 1:Q
                temp = bsxfun(@minus,data,mu_ref(q,:)');
                dist_frame = sqrt(sum(temp.^2)); % 1xT vector
                dist(p,q,n) = mean(dist_frame);
            end
        end        
    else
        % compute pairwise distance between cluster centers of current
        % sequence and the reference sequence
        for p = 1:Q
            for q = 1:Q                
                dist(p,q,n) = norm(C(p,:)-mu_ref(q,:));
            end
        end
        % Use Hungarian algorithm to solve optimal assignment for current
        % sequence w.r.t. the first sequence
        [assignment, cost] = assignmentoptimal(dist(:,:,n));
        % change the mapping of cluster assignment for current sequence        
        IDX_re = IDX;
        for ii=1:Q
            if assignment(ii)==0 
                fprintf('empty cluster %d in sequence %d\n',assignment(ii),n);
            else
                IDX_re(IDX==ii) = assignment(ii);
            end
        end
        IDX = IDX_re;        
    end
    IDX_all(first_slice_idx(n):crt_first_slice_idx-1) = IDX;
end
first_slice_idx(N+1) = crt_first_slice_idx;
IDX_all = IDX_all(1:crt_first_slice_idx-1);
data_all = data_all(:,1:crt_first_slice_idx-1);

% initialization
prior_all_count = zeros(Q,1);
transmat_all_count = zeros(Q,Q);
duramat_all_count = zeros(Q,L);
% shared by all sequences
mu = zeros(O,Q); 
Sigma = zeros(O,O,Q);
kappa = zeros(O,Q);
for j = 1:Q
    [mu(:,j), Sigma(:,:,j), mix] = mixgauss_init(1, data_all(:,IDX_all==j), cov_type);
    if size(cov_prior,3) == 1
        Sigma(:,:,j) = Sigma(:,:,j) + cov_prior; %%%%%%%%%%%% to prevent too concentrated data distribution
    else
        Sigma(:,:,j) = Sigma(:,:,j) + cov_prior(:,:,j);
    end
    [~,p] = chol(Sigma(:,:,j));
    if p>0
        Sigma(:,:,j) = 100*eye(O);
        fprintf('initial covariance of state %d mixture %d is not psd, use default initialization\n',j,M);
    end
    kappa(:,j) = 0.5*log(diag(Sigma(:,:,j)));
end
% get statistics of prior, transition and duration
% based on IDX of each sequence
for n = 1:N
    IDX_n = IDX_all(first_slice_idx(n):first_slice_idx(n+1)-1);    
    prior_all_count(IDX_n(1)) = prior_all_count(IDX_n(1)) + 1;
    [~,C] = est_transmat(IDX_n,Q);
    transmat_all_count = transmat_all_count + C;
    %~ need to ensure each sequence contains all different states
    [~,C] = est_duramat(IDX_n,Q,L);
    duramat_all_count = duramat_all_count + C;
end
prior = prior_all_count/N; % all sequences share the same initial state distribution
transmat_all_count = transmat_all_count - diag(diag(transmat_all_count));
transmat = mk_stochastic(transmat_all_count);
idx = find(diag(transmat) > 0, 1);
if ~isempty(idx)
    warning('absorbing state which does not transit to a different state');
    % forbid self-transition
    transmat = transmat - diag(diag(transmat));
    transmat = mk_stochastic(transmat); % uniform among all other states except the self state
end

if strcmp(dura_type,'Multinomial')
    duramat = mk_stochastic(duramat_all_count + dura_prior); % QxL
elseif strcmp(dura_type,'Poisson')
    duramat = (duramat_all_count*([0:L-1]')+dura_prior(:,1))./(sum(duramat_all_count,2)+dura_prior(:,2)); % Qx1, dura follows Poisson distribution, dura starts from 0, we shift to start from 1
else
    error('Unsupported duration distribution.')
end

params_ini.prior = prior;
params_ini.transmat = transmat; 
params_ini.duramat = duramat;
params_ini.kappa_C = log(duramat);
params_ini.mu = mu;
params_ini.sigma = Sigma;
params_ini.mixmat = ones(Q,M)/M;
params_ini.kappa = kappa;

end
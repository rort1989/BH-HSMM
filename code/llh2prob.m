function prob = llh2prob(llh)
% given llh of each class, compute the probability belonging to each class,
% assuming uniform prior on label P(y) = 1/K

%[N,K] = size(llh); % N instances, K possible classes
denom = logsumexp(llh,2); % N x 1
temp = bsxfun(@minus,llh,denom);
prob = exp(temp);
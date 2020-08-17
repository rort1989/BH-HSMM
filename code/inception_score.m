function is = inception_score(prob)
% compute the inception score based on the class conditional probabilities
% reference: Salimans et al. Improved Techniques for Training GANs, 2016
[N,K] = size(prob);
KL1 = prob.*log(prob+eps); % NxK
KL2 = prob.*repmat(log(mean(prob)+eps),N,1); % 1xK
KL = sum(KL1-KL2,2); % Nx1
is = exp(mean(KL)); % scalar
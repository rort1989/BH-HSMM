function [grad_pi, grad_A, grad_kappa_C, grad_mu, grad_kappa, flag] = gradient_HSMM(x, prior, transmat, duramat, mu, sigma, mixmat)
% function to compute gradient of parameters in HSMM
% explicit duration HMM, assume no-self transition, Poisson duration
% distribution 
% x_g: DxT matrix
% a more efficient implementation than gradient_HSMM_
% Reference:
% Rui Zhao, Dynamic Data Modeling, Recognition, and Synthesis, 2018, Ph.D.
% dissertation, Appendix C

[O,T] = size(x);
Q = size(mu,2);
obslik = mixgauss_prob(x, mu, sigma, mixmat); % f_q: QxT
%%
phi = zeros(Q,T+1,T); % each entry is a number between 0 and 1
denom = [1 cumprod(1:T-2)]; % 1xT-1
mult = exp(-duramat); % Qx1 duramat is the lambda parameter in Poisson distribution
for q = 1:Q    
    temp = duramat(q).^[0:T-2]; % 1xT-1
    temp = temp*mult(q)./denom;
    phi(q,1:T-1,1) = temp;
    phi(q,T,1) = max(0,1 - sum(temp));
end
Cd = phi(:,1:T,1); % P(D=t|Z=q) % QxT
phi(:,1:T,1) = phi(:,1:T,1).*repmat(prior,1,T); % QxT

%%
cc = zeros(1,T); % normalization constant
cc(1) = obslik(:,1)'*prior; % 1xQ * Qx1 -> 1
% initial state, page 3
grad_pi = obslik(:,1) / cc(1); % Qx1
grad_A = zeros(Q,Q); % QxQ, forbid self-transition i.e. diagonal entries are 0
grad_kappa_C = zeros(Q,1); % Qx1, Poisson parameters, reparameterize to ensure positive
grad_mu = zeros(O,Q);
grad_kappa = zeros(O,Q); % variance parameters, reparameterize to ensure positive

% transition
grad_phi_A = zeros(Q,Q,Q,T+1,T); % first two dimensions is the dimension of gradient, the initial grad at t=1 is 0, the last slice in the second last dimension is always 0
% duration
grad_phi_kappa_C = zeros(Q,Q,T+1,T); % first dimension is the dimension of gradient
% emission: ONLY support diagonal covariance
grad_f_mu = zeros(O,Q,T);
grad_f_kappa = zeros(O,Q,T); % kappa = log(sigma), only diagonal covariance only
grad_phi_mu = zeros(O,Q,Q,T+1,T); % the initial grad at t=1 is 0
grad_phi_kappa = zeros(O,Q,Q,T+1,T); % the initial grad at t=1 is 0

if cc(1) == 0 % the data is way off, don't compute its gradient anymore
    grad_pi = zeros(Q,1);
    flag = true;

else
    % compute grad_f_mu and grad_f_kappa for all t, reuse the same as HMM
    for q = 1:Q
        x_mu = bsxfun(@minus,x,mu(:,q)); % OxT
        inv_sigma_mu = sigma(:,:,q)\x_mu; % OxT
        grad_f_mu(:,q,:) = bsxfun(@times,inv_sigma_mu,obslik(q,:)); % OxT .* 1xT -> OxT
        grad_f_kappa(:,q,:) = bsxfun(@times,(inv_sigma_mu.*x_mu)-1,obslik(q,:)); % OxT .* 1xT
        % compute grad of phi_kappa_C for t = 1
        grad_phi_kappa_C(q,q,1:T,1) = ([0 Cd(q,1:T-1)] - Cd(q,:))*duramat(q); % 1xT
    end

    % some constant vector or matrix
    e = ones(Q) - eye(Q);
    z = zeros(Q,T,Q);
    for q = 1:Q
        z(q,:,q) = 1;
    end

    %% global gradient at t=1, page 2
    t = 1;
    term_C = sum(grad_phi_kappa_C(:,:,:,t),3) * obslik(:,t); % QxQ * Qx1 -> Qx1
    grad_kappa_C = term_C/cc(t); % Qx1

    term_mu_1 = bsxfun(@times, grad_f_mu(:,:,t), sum(phi(:,1:T,t),2)'); % OxQ .* 1xQ -> OxQ
    grad_mu = term_mu_1/cc(t); % OxQ

    term_kappa_1 = bsxfun(@times, grad_f_kappa(:,:,t), sum(phi(:,1:T,t),2)'); % OxQ .* 1xQ -> OxQ
    grad_kappa =  term_kappa_1/cc(t); % OxQ

    %% recursion to compute local gradient at each t and accumulate global gradient
    for t = 2:T
        % recursively compute phi: page 1, both Q and T are taken care of using
        % matrix-vector multiplication, no loop needed
        obs_phi = obslik(:,t-1).*phi(:,1,t-1); % Qx1
        mult = transmat'*obs_phi; % Qx1
        phi(:,1:T-1,t) = bsxfun(@times,Cd(:,1:T-1),mult) + bsxfun(@times,phi(:,2:T,t-1),obslik(:,t-1)); % Qx(T-1)
        phi(:,T,t) = Cd(:,T).*mult; % Qx1
        phi(:,1:T,t) = phi(:,1:T,t)/cc(t-1); % QxT

        % use obslik and phi to compute c_t
        cc(t) = obslik(:,t)'*sum(phi(:,1:T,t),2);
        % assert(phi(:,T+1,t)==0)

        % compute local gradient: 
        %% transition:A page 4, duration:kappa_C page 5, emission:mu,kappa page 6
        % some constants shared across j
        sum_grad_phi_A = bsxfun(@minus, grad_phi_A(:,:,:,2:T+1,t-1), sum(grad_phi_A(:,:,:,2:T,t-1),4)); % QxQxQxT .- QxQxQ (the first: grad_phi_A(:,:,:,T+1,t-1)=0)
        sum_grad_phi_kappa_C = bsxfun(@minus, grad_phi_kappa_C(:,:,2:T+1,t-1), sum(grad_phi_kappa_C(:,:,2:T,t-1),3)); % QxQxT .- QxQ (the first: grad_phi_kappa_C(:,:,T+1,t-1)=0)
        sum_phi = bsxfun(@minus, phi(:,2:T+1,t-1), sum(phi(:,2:T,t-1),2)); % QxT        
        sum_grad_phi_mu = bsxfun(@minus, grad_phi_mu(:,:,:,2:T+1,t-1), sum(grad_phi_mu(:,:,:,2:T,t-1),4)); % OxQxQxT .- OxQxQ (the first: grad_phi_mu(:,:,:,T+1,t-1)=0)
        sum_grad_phi_kappa = bsxfun(@minus, grad_phi_kappa(:,:,:,2:T+1,t-1), sum(grad_phi_kappa(:,:,:,2:T,t-1),4)); % OxQxQxT .- OxQxQ (the first: grad_phi_mu(:,:,:,T+1,t-1)=0)

        for j = 1:Q
            phi_rep = repmat(phi(j,1:T,t),Q,1);        

            mult1 = transmat(:,j)*Cd(j,:) - phi_rep; % QxT
            mult1_obs = bsxfun(@times,mult1,obslik(:,t-1)); % QxT
            mult1_obs_reshape = reshape(mult1_obs,[1,1,Q,T]);

            term1_1_A = bsxfun(@times, mult1_obs_reshape, grad_phi_A(:,:,:,1,t-1)); % 1x1xQxT .* QxQxQ -> QxQxQxT        
            term1_1_C = grad_phi_kappa_C(:,:,1,t-1) * mult1_obs; % QxQ * QxT -> QxT

            term1_2_A = zeros(Q,Q,T);
            term1_2_A(:,j,:) = (e(:,j) .* obs_phi) *Cd(j,:); % QxQxT, only QxT are non-zero

            mult1_2_C = ([0 Cd(j,1:T-1)] - Cd(j,:))*duramat(j); % 1xT

            term1_2_C = zeros(Q,T);
            term1_2_C(j,:) = sum(obs_phi.*transmat(:,j)) * mult1_2_C; % QxT, only 1xT are non-zero        

            mult2 = z(:,:,j) - phi_rep; % QxT
            mult2_obs = bsxfun(@times,mult2,obslik(:,t-1)); % QxT
            mult2_obs_reshape = reshape(mult2_obs,[1,1,Q,T]);

            term2_A = bsxfun(@times, mult2_obs_reshape, grad_phi_A(:,:,:,2:T+1,t-1)); % 1x1xQxT, QxQxQxT -> QxQxQxT
            term2_C = bsxfun(@times, reshape(mult2_obs,[1,Q,T]), grad_phi_kappa_C(:,:,2:T+1,t-1)); % 1xQxT, QxQxT -> QxQxT

            mult3_obs = obslik(:,t-1)*phi(j,1:T,t); % QxT
            mult3_obs_reshape = reshape(mult3_obs,[1,1,Q,T]);

            term3_A = bsxfun(@times, mult3_obs_reshape, sum_grad_phi_A); % 1x1xQxT .* QxQxQxT -> QxQxQxT

            term3_C = bsxfun(@times, reshape(mult3_obs,[1,Q,T]), sum_grad_phi_kappa_C); % 1xQxT .* QxQxT -> QxQxT
            % 
            grad_phi_A(:,:,j,1:T,t) = (reshape(term1_2_A,[Q,Q,1,T]) + sum(term1_1_A,3) + sum(term2_A,3) + sum(term3_A,3))/cc(t-1); % QxQx1xT
            grad_phi_kappa_C(:,j,1:T,t) = (reshape(term1_1_C,[Q,1,T]) + reshape(term1_2_C,[Q,1,T]) + sum(term2_C,2) + sum(term3_C,2))/cc(t-1); % QxT
            %
            % assert(grad_phi_A(:,:,:,T+1,:)==0)
            % assert(grad_phi_kappa_C(:,:,T+1,:)==0)

            % mu
            mult1_1 = bsxfun(@times,mult1,phi(:,1,t-1)); % QxT  

            term1_1_mu = bsxfun(@times, reshape(mult1_1,[1,Q,T]), grad_f_mu(:,:,t-1)); % 1xQxT .* OxQ -> OxQxT
            term1_2_mu = bsxfun(@times, mult1_obs_reshape, grad_phi_mu(:,:,:,1,t-1)); % 1x1xQxT .* OxQxQ -> OxQxQxT
            term1_mu = reshape(term1_1_mu,[O,Q,1,T]) + sum(term1_2_mu,3); % OxQxT

            mult2_1 = mult2 .* phi(:,2:T+1,t-1); % QxT

            term2_1_mu = bsxfun(@times, reshape(mult2_1,[1,Q,T]), grad_f_mu(:,:,t-1)); % 1xQxT .* OxQ -> OxQxT
            term2_2_mu = bsxfun(@times, mult2_obs_reshape, grad_phi_mu(:,:,:,2:T+1,t-1)); % 1x1xQxT .* OxQxQxT -> OxQxQxT
            term2_mu = reshape(term2_1_mu,[O,Q,1,T]) + sum(term2_2_mu,3); % OxQxT

            mult3_1 = phi_rep .* sum_phi; % QxT

            term3_1_mu = bsxfun(@times, reshape(mult3_1,[1,Q,T]), grad_f_mu(:,:,t-1)); % 1xQxT .* OxQ -> OxQxT
            term3_2_mu = bsxfun(@times, mult3_obs_reshape, sum_grad_phi_mu); % 1x1xQxT .* QxQxQxT -> QxQxQxT
            term3_mu = reshape(term3_1_mu,[O,Q,1,T]) + sum(term3_2_mu,3); % OxQxT

            grad_phi_mu(:,:,j,1:T,t) = (term1_mu + term2_mu + term3_mu) / cc(t-1); % OxQxT

            % kappa
            term1_1_kappa = bsxfun(@times, reshape(mult1_1,[1,Q,T]), grad_f_kappa(:,:,t-1)); % 1xQxT .* OxQ -> OxQxT
            term1_2_kappa = bsxfun(@times, mult1_obs_reshape, grad_phi_kappa(:,:,:,1,t-1)); % 1x1xQxT .* OxQxQ -> OxQxQxT
            term1_kappa = reshape(term1_1_kappa,[O,Q,1,T]) + sum(term1_2_kappa,3); % OxQxT

            term2_1_kappa = bsxfun(@times, reshape(mult2_1,[1,Q,T]), grad_f_kappa(:,:,t-1)); % 1xQxT .* OxQ -> OxQxT
            term2_2_kappa = bsxfun(@times, mult2_obs_reshape, grad_phi_kappa(:,:,:,2:T+1,t-1)); % 1x1xQxT .* OxQxQxT -> OxQxQxT
            term2_kappa = reshape(term2_1_kappa,[O,Q,1,T]) + sum(term2_2_kappa,3); % OxQxT

            term3_1_kappa = bsxfun(@times, reshape(mult3_1,[1,Q,T]), grad_f_kappa(:,:,t-1)); % 1xQxT .* OxQ -> OxQxT
            term3_2_kappa = bsxfun(@times, mult3_obs_reshape, sum_grad_phi_kappa); % 1x1xQxT .* QxQxQxT -> QxQxQxT
            term3_kappa = reshape(term3_1_kappa,[O,Q,1,T]) + sum(term3_2_kappa,3); % OxQxT

            grad_phi_kappa(:,:,j,1:T,t) = (term1_kappa + term2_kappa + term3_kappa) / cc(t-1); % OxQxT

            % assert(grad_phi_mu(:,:,:,T+1,:)==0)
            % assert(grad_phi_kappa(:,:,:,T+1,:)==0)
        end

        %% compute sequence gradient: page 2
        obslik_reshape = reshape(obslik(:,t),[1,1,Q]);

        term_A = bsxfun(@times, sum(grad_phi_A(:,:,:,:,t),4), obslik_reshape); % QxQxQ
        grad_A = grad_A + sum(term_A,3)/cc(t); % QxQ
        % assert(norm(diag(grad_A))) = 0

        term_C = sum(grad_phi_kappa_C(:,:,:,t),3) * obslik(:,t); % QxQ * Qx1 -> Qx1
        grad_kappa_C = grad_kappa_C + term_C/cc(t); % Qx1

        term_mu_1 = bsxfun(@times, grad_f_mu(:,:,t), sum(phi(:,1:T,t),2)'); % OxQ .* 1xQ -> OxQ
        term_mu_2 = bsxfun(@times, sum(grad_phi_mu(:,:,:,:,t),4), obslik_reshape); % OxQxQ .* 1x1xQ -> OxQxQ
        term_mu = term_mu_1 + sum(term_mu_2,3);
        grad_mu = grad_mu + term_mu/cc(t); % OxQ

        term_kappa_1 = bsxfun(@times, grad_f_kappa(:,:,t), sum(phi(:,1:T,t),2)'); % OxQ .* 1xQ -> OxQ
        term_kappa_2 = bsxfun(@times, sum(grad_phi_kappa(:,:,:,:,t),4), obslik_reshape); % OxQxQ .* 1x1xQ -> OxQxQ
        term_kappa = term_kappa_1 + sum(term_kappa_2,3);
        grad_kappa =  grad_kappa + term_kappa/cc(t); % OxQ

    end

    flag = ~isempty(find(cc==0,1)) | isnan(sum(cc));
end

%~ debug: PASSED
% aa=grad_phi_A(:,:,:,T+1,:);
% bb=grad_phi_kappa_C(:,:,T+1,:);
% cc=grad_phi_mu(:,:,:,T+1,:);
% dd=grad_phi_kappa(:,:,:,T+1,:);
% assert(min(aa(:))==0&&max(aa(:))==0)
% assert(min(bb(:))==0&&max(bb(:))==0)
% assert(min(cc(:))==0&&max(cc(:))==0)
% assert(min(dd(:))==0&&max(dd(:))==0)
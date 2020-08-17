% script to train HDM and its variants given data and configuration of the
% experiment: unsupervised learning using all the unlabeled data
rng default;
num_batch = ceil(N/batch_size);
class_all = unique(labels_train(:,1));
K = length(class_all);

% Initialization of generator and discriminator
if reload > 0 % train from existing checkpoint
    filename = fullfile(savedir,sprintf('hdm_dataset%d_class%d_useale%d_syn%d_O%d_Q%d_iter%d.mat',dataset_id,class,useale,syn,O,sum(Q),reload));
    load(filename);
    params_gen_sets_ = params_gen_sets;
    params_gen_sets = repmat(struct('params',[]),1,(max_iters+reload)*num_batch);
    params_gen_sets(1:reload*num_batch) = params_gen_sets_;
    clear params_gen_sets_
else % train from scratch
    % generator is HDM with some fixed hyperparameters   
    [mu0, sigma0] = est_hyper_emis(dataset_train_r,cov_type,T*size(labels_train,1));
    inv_sigma_0 = mu0./diag(sigma0);
    hyperparams_dis.mu0 = repmat(mu0,1,Qd); % OxQ, can also do class-wise initialization
    hyperparams_dis.inv_sigma_0 = repmat(inv_sigma_0,1,Qd); % OxQ
    hyperparams_dis.theta = T/2;
    hyperparams_gen.mu0 = repmat(mu0,1,sum(Q)); % OxQ, can also do class-wise initialization
    hyperparams_gen.inv_sigma_0 = repmat(inv_sigma_0,1,sum(Q)); % OxQ
    hyperparams_gen.theta = T/2;
    params_ini_all = repmat(struct('params',[]),1,K);
    for k = 1:K  %%%%%%%%%%% use a different cov_prior than discriminator
        % use evenly split state: Q/K
        params_ini_all(k).params = ini_HSMM(dataset_train_r(labels_train(:,1)==class_all(k)), Q(k), O, 1, T, 'cov_type', cov_type, 'cov_prior', 10*cov_prior, 'dura_type', dura_type, 'dura_prior', [0,0]);
    end
    % use evenly split state: Q/K*ones(1,K)
    params_gen = combine_params(params_ini_all,Q,O,1,dura_type);
    params_gen_sets = repmat(struct('params',[]),1,max_iters*num_batch);
    
    % discriminator contains a set of HSMMs, one for each class
    % initialize all of them as the same parameters that learned from
    % corresponding class of real data
    params_dis = repmat(struct('params',[]),1,K);
    for k = 1:K % ini_hsmm_
        params_dis(k).params = ini_HSMM(dataset_train_r(labels_train(:,1)==class_all(k)), Qd, O, 1, T, 'cov_type', cov_type, 'cov_prior', cov_prior, 'dura_type', dura_type, 'dura_prior', [0,0]);
    end
    V_d = cell(K,1);
    for k = 1:K
        % initialize value of RMS of accumulate gradient
        [V_d{k},V_g] = ini_RMS_HSMM(sum(Q),O,Qd);
    end
end
fprintf('Initialization completed.\n');
% main loop
id_perm = randperm(N);
for it = 1+reload:max_iters+reload  %%%%%%%%%%%%%%%%%%%%%% later on modify this to use epoch
    % each iteration should go through an epoch
    tt = tic;
    % learning rate decay
    if mod(it-1,decay_epoch)==0
        ff = decay_d^round((it-1)/decay_epoch);
        Rd = RHO_d*ff*ones(1,5);
        Rd([3,5]) = Rd([3,5])/10; % smaller learning rate for log(lambda) and log(sigma)
        ff = decay_g^round((it-1)/decay_epoch);
        Rg = RHO_g*ff*ones(1,5);
        Rg([3,5]) = Rg([3,5])/10; % smaller learning rate for log(lambda) and log(sigma)
    end

    % update based on each batch
    for ot = 1:num_batch
        fprintf('Iteration %d: ',it);
        % update discriminator
        % step D0: draw samples from real data and generator
        % select a random subset of dataset_train_r
        idx_batch = id_perm(batch_size*(ot-1)+1:min(batch_size*ot,N));
        dataset_train_r_batch = dataset_train_r(idx_batch);
        dataset_train_g_batch = sample_HSMM(params_gen,batch_size,T,1);
        for k = 1:k_step_D
            % step D0: compute coefficient %%%% consider other correction value
            [en_r,en_g,prob_r,prob_g] = coefficient_discriminator_HSMM_un(dataset_train_r_batch,dataset_train_g_batch,params_dis,T,dura_type);
            % step D1: compute positive model gradient
            % consider using RMSprop and other adaptive gradient method
            GRAD = grad_discriminator_un_HSMM(params_dis,dataset_train_r_batch,dataset_train_g_batch,en_r,en_g,prob_r,prob_g);        
            % consider using RMSprop and other adaptive gradient method
            for l = 1:K                
                GRAD{l} = updateGradient_HSMM_SGHMC(GRAD{l},params_dis(l).params,hyperparams_dis,alpha,batch_size);
                % clip the gradient: set to 0 if GRAD is NaN or too large/small
                GRAD{l} = grad_clip(GRAD{l},clip_thresh,clip_val);
                assert(norm(diag(GRAD{l}{2}))==0)
                % step D2: update model parameters
                [params_dis(l).params,V_d{l}] = updateParams_HSMM_SGHMC(params_dis(l).params,GRAD{l},Rd,V_d{l},alpha);
            end
        end
        % collect results
        history.en_r(ot,it) = -mean(en_r(en_r<0));
        history.en_g(ot,it) = -mean(en_g(en_g<0));
        history.loss_d(ot,it) = -mean(en_r(en_r<0)) + mean(en_g(en_g<0)); % negation of objective as we are doing Gradient ascent
        fprintf('D-batch %d, loss: %f, ',ot,history.loss_d(ot,it));
        
        % update generator
        % step G0: draw fake samples
        dataset_train_g_batch = sample_HSMM(params_gen,batch_size,T,1);
        for k = 1:k_step_G
            % step G0: compute coefficient
            [g,prob,llh] = coefficient_generator_HSMM_un(dataset_train_g_batch,params_dis,T,dura_type);
            % step G1: compute gradient
            [GRAD, idx_valid] = grad_generator_HSMM(params_gen,dataset_train_g_batch,g);           
            % consider using RMSprop and other adaptive gradient method
            GRAD = updateGradient_HSMM_SGHMC(GRAD,params_gen,hyperparams_gen,alpha,batch_size);
            % clip the gradient: set to 0 if GRAD is NaN or too large/small
            GRAD = grad_clip(GRAD,clip_thresh,clip_val);
            assert(norm(diag(GRAD{2}))==0)
            % step G2: update parameters
            [params_gen,V_g] = updateParams_HSMM_SGHMC(params_gen,GRAD,Rg,V_g,alpha);
        end
        % collect results
        history.loss_g(ot,it) = -mean(g(idx_valid)); % negation of objective
        fprintf('G-batch %d, loss: %f',ot,history.loss_g(ot,it));
        fprintf('\n');
        if isnan(history.loss_g(ot,it))
           disp('debug')
        end
        
        % collect samples of generator
        params = params_gen;
        params_gen_sets((it-1)*num_batch+ot).params = params;
    end
    
    time(it) = toc(tt);
    fprintf('Iteration %d completed.\n',it);
    
    % save model every few iterations
    if mod(it,savestep) == 0
        save(fullfile(savedir,sprintf('hdm_dataset%d_class%d_useale%d_syn%d_O%d_Q%d_iter%d',dataset_id,class,useale,syn,O,sum(Q),it)),'params_gen','params_gen_sets','history','params_ini_all','params_dis','time','V_d','V_g','hyperparams_dis','hyperparams_gen'); %,'RMS_d','RMS_g' ,'c1','c2'
    end
    
end


%% plot the overall loss change
if max_iters > 0
    figure;
    subplot(2,1,1)
    plot(1:it*num_batch, history.loss_d(:),'b',1:it*num_batch, history.en_r(:),'r', 1:it*num_batch, history.en_g(:), 'g')%
    legend({'discriminator loss','avg. entropy of real data','avg. entropy of fake data'})
    subplot(2,1,2)
    plot(1:it*num_batch, history.loss_g(:))
    legend({'generator loss'})
    savefig(fullfile(savedir,sprintf('hdm_dataset%d_loss.fig',dataset_id)))
else
    history = [];
end

if llh_train == 1 % learned model loglikelihood on training data
    % a better way is to find MAP estimate for each data and then compute its likelihood    
    [loglikelihood, viterbicode] = compute_llh_evidence_HSMM(dataset_train_r, params_gen, T, 'dura_type', dura_type);
else
    loglikelihood = [];
    viterbicode = [];
end
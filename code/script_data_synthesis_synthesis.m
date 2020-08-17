% script to generate synthetic data with ancestral sampling using learned
% dynamic model
rng default;
usernd = 0; % exploration, see what is the quality of using random sampled observations

if dataset_id == 1
    load(fullfile(dir_base,'CMUMocap','01.mat'),'skel');
    njoint = length(skel.tree);
    topo = [0:5 1 7:10 1 12:16 14 18:22 21 14 25:29 28]';
elseif dataset_id == 2
    load(fullfile(dir_base,'Berkeley','skel01.mat'),'skel');
    njoint = length(skel.tree);
    topo = [0:6 4 8:13 4 15:20 1 22:27 1 29:34]';
end
num_class = length(class);
T_ = T-1;
dataset_syn = cell(N_,1); % zeros(O/2,T_,N_);
llh = zeros(N_,1);

% ancestral sampling for data generation
state = cell(N_,1);
% if viterbi % decode training data and generate synthetic conditioned on training state
%     viterbicode = viterbi_path_partial(dataset_train_r,params_gen,[]);
% else
%     viterbicode = cell(N_,1);
% end
if useale == 1
    [dataset,params_g] = sample_hdm(params_gen,N_,1,T_); % num_params = N_
    for j = 1:N_ % sample N_ set of parameters
        q = sample_discrete(params_g(j).prior);
        state{j} = sample_path(q,params_g(j).transmat,params_g(j).duramat,T_); % optimal_path
        dataset_syn{j} = estimate_Position(params_g(j), state{j}); % O/2
        %llh(j) = compute_llh_evidence_HMM(dataset_syn{j}, params_g(j), 0);
    end
elseif useale == 4
    % potentially multiple set of parameters obtained through Bayesian
    % inference
    Nset = length(params_gen_sets);
    norm_c = repmat(cumprod([1 1:T_-1]),sum(Q),1);
    for j = 1:N_ % sample N_ set of parameters
        if ~isempty(params_gen_sets)
            params_g = params_gen_sets(mod(j-1,Nset)+1).params;
        else
            params_g = params_gen;
        end
        if viterbi
            temp = params_g.duramat(:,ones(1,T_)).^repmat(0:T_-1,sum(Q),1) .* exp(-params_g.duramat(:,ones(1,T_))) ./ norm_c;
            duramat_table = mk_stochastic(temp);
            obslik = mixgauss_prob(dataset_train_r{mod(j-1,N)+1}, params_g.mu, params_g.sigma, params_g.mixmat);
            state{j} = viterbi_path_semi(params_g.prior, params_g.transmat, duramat_table, obslik);
            state{j} = state{j}(1:T_);
        else
            q = sample_discrete(params_g.prior);
            state{j} = sample_path(q,params_g.transmat,params_g.duramat,T_,1+(dataset_id==1)*(q>6)); % optimal_path
        end
        if usernd ~= 1
            dataset_syn{j} = estimate_Position(params_g, state{j}); % O/2
        else % sample data directly from conditional distribution
            for t = 1:T_
                dataset_syn{j}(:,t) = gaussian_sample(params_g.mu(1:O/2,state{j}(t)), params_g.sigma(1:O/2,1:O/2,state{j}(t)), 1);
            end
        end        
    end    
elseif useale == 2 % HDM initial
    for j = 1:N_ % sample N_ set of parameters
        q = sample_discrete(params_gen.prior);
        state{j} = sample_path(q,params_gen.transmat,params_gen.duramat,T_); % optimal_path
        dataset_syn{j} = estimate_Position(params_gen, state{j}); % O/2
    end
elseif useale == 0 % HMM
    dataset = synthetic_HMM(1,N_,T_,Q,O,'params',params_gen); % an even more naive synthesis
    for j = 1:N_ % sample N_ set of parameters    
        dataset_syn{j} = dataset.observed(1:O/2,:,j);
    end
end

%% postprocessing: filtering
figure;
if usefilter
    for j = 1:N_
        for o = 1:size(dataset_syn{j},1) % O/2
            temp = dataset_syn{j}(o,:);
            dataset_syn{j}(o,:) = anisotropic_diffusion_filter(temp,0.2,10,3);
        end
    end
    % checkout the filtering quality
    plot(1:T_,temp,1:T_,dataset_syn{j}(o,:),'r');
else
    plot(1:T_,dataset_syn{1}(1,:),'r');
end

% rescale the synthesized data
obs_scaled = scaleData_z_r(dataset_syn, matScale(idx_channel,:));

% augment to have constant channels and offset location
obs_scaled = augData(obs_scaled,idx_channel+3,[zeros(3,1); matScale(:,1)]);

% interpolation of each skeleton coordinate trajectories
if dataset_id == 1
    dataset_g_pos = angle2pos(obs_scaled,skel,njoint);
    displayrange = [-15 15 -15 15 -20 15];
else
    dataset_g_pos = angle2pos_b(obs_scaled,skel,njoint);
    displayrange = [-70 70 -30 70 -100 80];
end

% create video (without reference mean pose)
figure;
v = VideoWriter(fullfile(savedir,sprintf('sample_dataset%d_class%d_hdm%d_filter%d_ale%d_iters%d_vtb%d_syn.avi',dataset_id,class,trial,usefilter,useale,reload+max_iters,viterbi)));
v.FrameRate = 30;
open(v);
oneframe = zeros(504,672,3,'uint8');
samples = 1:N_/20:N_;
for n = 1:length(samples)
    for t = 1:T_
        drawskt3(dataset_g_pos{samples(n)}(1:3:end,t),dataset_g_pos{samples(n)}(3:3:end,t),dataset_g_pos{samples(n)}(2:3:end,t),1:njoint,topo,'displayrange',displayrange,'MarkerSize',20,'LineWidth',5);
        title(num2str(n));
        % view(-135,30)
        f = getframe(gcf);
        oneframe = f.cdata;%(1:504,1:672,:)
        writeVideo(v,oneframe);
    end
end
close(v);

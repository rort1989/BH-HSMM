% script to perform classification using HMM
% train HMM, one for each class (classification in the position domain)
num_class = length(class_all);
filename = fullfile(savedir,sprintf('dataset%d_class%d_syn%d.mat',dataset_id,class,syn));
rewrite = 1;  % set to 0 if reloading from last saved results
if rewrite
    clear acc_train params_set
else
    load(filename,'acc_train','params_set');
end

if ~exist('acc_train','var') % this part can be replaced by another classifier
    Qc = 3:15;
    dataset_train = cell(num_class,1);
    true_labels_train = [];
    for a = 1:num_class
        % idx of training
        idx = find(labels_train(:,1)==class_all(a));
        dataset_train{a} = cell(length(idx),1);
        for n = 1:length(idx)
            dataset_train{a}{n} = dataset_train_r{idx(n)}(1:end/2,:); % joint angles
        end
        true_labels_train = [true_labels_train; a*ones(length(dataset_train{a}),1)];
    end
    % data train and test all in one array of cells
    data_train_all = [];
    for a = 1:num_class
        data_train_all = [data_train_all; dataset_train{a}];
    end
    
    %%%%%%%%%%% training
    llh_train = zeros(length(data_train_all),num_class);
    params_set = repmat(struct('params',[]),num_class,1);
    Oc = size(data_train_all{1},1);
    acc_max = 0;
    
    for it = 1:length(Qc) % different random initialization
        tt = tic;
        for a = 1:num_class
            [params_set(a).params, ~, LLtrace] = learn_params_HMM_BNT(dataset_train{a},Qc(it),Oc,0,'ini',1,'cov_prior',repmat(0.01*eye(Oc),[1 1 Qc(it)]),'cov_type','diag','mixture',M,'max_iter',10); % use K-means center, continuous observations
            % compute likelihood for testing data
            [llh_train(:,a)] = compute_llh_evidence_HMM(data_train_all, params_set(a).params, 0, 'viterbi', 0);
            
            fprintf(sprintf('Class %d completed.\n',a));
        end
        %% training results
        [~,label_temp] = max(llh_train,[],2);
        acc_train(it) = sum(label_temp(:)==true_labels_train)/length(label_temp);
        [cmatrix_train(:,:,it), cmatrix_norm_train(:,:,it)] = cm(num_class,true_labels_train,label_temp);
        if acc_train(it) > acc_max
            params_set_opt = params_set;
            acc_max = acc_train(it);
            cmatrix_opt = cmatrix_norm_train(:,:,it);
            Q_opt = Qc(it);
        end
        time(it) = toc(tt);
    end
    drawcm(cmatrix_opt); title(sprintf('Q=%d',Q_opt));
    params_set = params_set_opt;
    % identify the best model and save it
    save(filename,'acc_train','params_set','Qc','cmatrix_train','cmatrix_norm_train','-append'); 
end

% compute likelihood for all the testing data
llh_test = zeros(length(dataset_g_pos),num_class);
for a = 1:num_class
    [llh_test(:,a)] = compute_llh_evidence_HMM(dataset_syn, params_set(a).params, 0, 'viterbi', 0); %dataset_g_pos
end
llh_test = llh_test/T_;

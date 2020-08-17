% script to compute results given result of each fold
% for synthesis this would be used to compute all the evaluation criteria
if experiment == 3

    %% inception_score = [];
    % call a classifier to classify synthesized data
    % classification
    script_data_synthesis_classification

    %% compute inception score
    prob_is = llh2prob(llh_test); % NxK
    iscore = inception_score(prob_is);
    
    nfold = 10;
    p = randperm(N_);
    prob = [];
    iscore_all = zeros(nfold,1);
    for f = 1:nfold
        iscore_all(f) = inception_score(prob_is(p((f-1)*N_/nfold+1:f*N_/nfold),:));
    end

    %% Follow instruction to install nltk and python (windows only)
    % 1. Install Python
    % 2. Let matlab know the python interpretor path
    % 	pyversion path/to/python/executable e.g. C:/Program Files/Python35
    % 3. Install optional python module in computer (e.g. command window) through pip
    % 	pip install --user numpy
    % 	Use --user to install for current user only
    % 4. Add the installed package path to matlab e.g.
    % 	insert(py.sys.path,int32(0),'C:\Users\zhaor\AppData\Roaming\Python\Python35\site-packages');
    % 5. To use python module in matlab, start with py + command
    % e.g. 
    % py.sys.path
    % py.importlib.import_module('numpy')
    % py.importlib.import_module('nltk.translate.bleu_score')

    % quantize the data into different visual words
    BLEUscore_test = [];
    rewrite = 1; % repeat quantization
    for KD = [5 10 15 20 25 30 35]
        % perform quantization
        script_data_synthesis_quantization

        % call py.nltk.translate.bleu to compute bleu score
        % pre-requisite: check google doc to setup matlab to call python modules
        py.importlib.import_module('nltk.translate.bleu_score');
        weight = 0.25*ones(1,4); % up to 4-gram
        % construct reference sequence for training data: for validation purpose, just need to run it once
        dataset_train_ref = cell(1,N);
        dataset_rnd = cell(1,N);
        for n = 1:N
            dataset_train_ref{n} = dataset_quant([1:n-1 n+1:N])';
            dataset_rnd{n} = randi(KD,1,T);
        end
        BLEUscore_train = py.nltk.translate.bleu_score.corpus_bleu(dataset_train_ref, dataset_quant', weight);
        BLEUscore_rnd = py.nltk.translate.bleu_score.corpus_bleu(dataset_train_ref, dataset_rnd, weight);
        % construct reference sequence for synthetic sequence
        dataset_quant_ = dataset_quant;
        for m = 1:N
            dataset_quant_{m} = dataset_quant{m}(1:T_);    
        end
        dataset_test_ref = cell(1,N_);
        for n = 1:N_
           dataset_test_ref{n} = dataset_quant_';
        end
        BLEUscore_test = [BLEUscore_test py.nltk.translate.bleu_score.corpus_bleu(dataset_test_ref, dataset_g_quant', weight)];
    end

    %% mmd = [];
    % parameters for variant 1
    alpha = 0.05; % does not affect testState
    params_.sig = -1;
    params_.shuff = 10;
    params_.bootForce = 1;

    %% divide synthesized data into different batches of the same number of real
    % training data
    % Op = size(dataset_train_r_pos{1},1);
    Op = O/2;
    Y = zeros(N,Op*T_);
    %dataset_train_r_pos_ = dataset_train_r_pos;
    dataset_train_r_pos_ = dataset_train_r;
    for n = 1:N
        dataset_train_r_pos_{n} = dataset_train_r_pos_{n}(1:Op,1:T_)'; % TxD
        Y(n,:) = dataset_train_r_pos_{n}(:)';
    end
    repeat = 10;
    idx = round(linspace(1,N_-N+1,repeat)); %%%%%% assume N_ > N
    testStat = zeros(repeat,3);
    time = zeros(repeat,3);

    for i = 1:length(idx)
        % Original Kernel two-sample test based MMD
        % Gretton et al. A Kernel Two-Sample Test, 2012
        tt = tic;
        X = zeros(N,Op*T_);
        %dataset_g_pos_ = cell(N,1);
        for n = 1:N % idx(i):idx(i)+N-1
            X(n,:) = dataset_syn{idx(i)-1+n}(:)'; 
            % X(n,:) = dataset_g_pos{idx(i)-1+n}(:)';
            % dataset_g_pos_{n} = dataset_syn{n}'; 
            % dataset_g_pos_{n} = dataset_g_pos{n}';
        end
        [testStat(i,1),thresh] = mmdTestBoot(X,Y,alpha,params_);
        time(i,1) = toc(tt);

        % Non-parametric, low variance kernel two-sample tests
        % Zaremba et al. B-tests: Low Variance Kernel Two-Sample Tests, 2013
        tt = tic;
        [~, p2,testStat(i,2)] = btest(num2cell(X,2), num2cell(Y,2));
        time(i,2) = toc(tt);

        % Wild bootstrap tests for time series
        % Chwialkowski et al. Wild bootstrap Kernel Tests, 2014
        tt = tic;
        mmd3 = wildMMD(X,Y); % (dataset_g_pos_, dataset_train_r_pos_)
        testStat(i,3) = mmd3.testStat;
        time(i,3) = toc(tt);
    end

    %% ssim = [];
    % addpath('C:\Users\zhaor\Dropbox\tools\SSIM');
    % [mssim, ssim_map] = ssim(img1, img2, K, window, L);

    %% others
    results.llh_all = [];
    results.llh_avg = [];
    results.is = mean(iscore_all);
    results.is_std = std(iscore_all);
    results.is_all = iscore_all;
    results.bleu = BLEUscore_test;
    results.mmd = mean(testStat);
    results.mmd_std = std(testStat);
    results.mmd_testStat = testStat;

else
    results.PCC = PCC;
    results.MSE = MSE;
    results.PCC_avg = mean(PCC(:));
    results.MSE_avg = mean(MSE(:));
end
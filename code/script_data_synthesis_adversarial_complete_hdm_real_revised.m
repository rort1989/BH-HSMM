%% Adversarial learning program with for data synthesis
clear all; dbstop if error; close all;

%% add dependency
addpath('../tools/imCRBM-master/Motion');
addpath('../tools/relabeler');
addpath('../tools/MOCAP0p136');
addpath('../tools/bnt');
addpath(genpathKPM('../tools/bnt'))
addpath('../tools/mmd');
addpath('../tools/mmd_btest');
addpath('../tools/mmd_wildbootstrap');

%% set experiment configuration
class = 99;
syn = 1;
experiment = 3; % 1. reconstruct left leg joint angles; 2. reconstruct right arm joint angles; 3. synthesis; 4. prediction
dataset_id = input('Enter dataset number:\n [1]CMU\n [2]Berkely\n'); % choose which dataset to experiment with
if isempty(dataset_id) || dataset_id<1 || dataset_id>2
    error('Unsupported dataset. Enter a number between 1 and 2.');
end

%% load real data
savedir = '../results/synthesis';
dir_base = '../data';
filename = fullfile(savedir,sprintf('dataset%d_class%d_syn%d.mat',dataset_id,class,syn));
if ~exist(filename,'file') % load saved data if exists
    if dataset_id == 1
        script_data_synthesis_preparedataset_CMU_revised; % after this we have: dataset_scaled, T, O, N, stride, config, subjects
    elseif dataset_id == 2
        script_data_synthesis_preparedataset_Berkeley_revised;
    end
    save(filename,'T','O','N','matScale','idx_channel','dataset_pos','dataset_scaled','labels','njoint','topo','class_all');
else
    load(filename,'T','O','N','matScale','idx_channel','dataset_pos','dataset_scaled','labels','njoint','topo','class_all');
end

%% Define learning configuration e.g. SGD hyperparameters
if dataset_id == 1
    configuration_synthesis_CMU;
elseif dataset_id == 2
    configuration_synthesis_Berkeley;
end

%% split training and testing data
if dataset_id == 1
    idx = [1:40 41 42:5:133 134:137];% []; %choose to balance different classes
elseif dataset_id == 2
    idx = [1:2:120 121:3:289 290:7:720 721:4:939 940:4:1162 1163:4:1398 1399:2:1498 1499:1691];
end
dataset_train_r = dataset_scaled(idx);
dataset_train_r_pos = dataset_pos(idx);
labels_train = labels(idx,:);
dataset_test_r = dataset_train_r;
N = length(dataset_train_r);
T = size(dataset_train_r{1},2);
% optional: plot the data
% for n = 1%:N
%     drawskt3(dataset_train_r_pos{n}(1:3:end,:),dataset_train_r_pos{n}(3:3:end,:),dataset_train_r_pos{n}(2:3:end,:),1:njoint,topo);
%     fprintf('%d out of %d is plot\n',n,N);
% end

%% training: major part
filename = fullfile(savedir,sprintf('hdm_dataset%d_class%d_useale%d_syn%d_O%d_Q%d_iter%d.mat',dataset_id,class,useale,syn,O,sum(Q),(reload+max_iters)*(useale==1|useale==4)));
% filename = fullfile(savedir,sprintf('hdm_dataset%d_class%d_M%d_O%d_Q%d_T%d_ale%d_exp%d_trial%d_iter%d.mat',dataset_id,class,M,O,sum(Q),T-1,useale,3,trial,(reload+max_iters)*(useale==1|useale==4)));
if exist(filename,'file') % load trained model if exists
    load(filename,'params_gen','params_dis','history','params_gen_sets','hyperparams_gen');
    params_ini = params_gen;
    loglikelihood = []; time = [];
else % otherwise train model
    train_hdm_un_Bayesian
end

%% evaluation: see whether the discriminator cannot tell the difference of generated data versus real data or not
script_data_synthesis_plot;

%% perform experiment: data synthesis
script_data_synthesis_synthesis;

%% analyze results
script_data_synthesis_evaluation;

%% some of the entries will not exist if you try to re-run experiment using existing models
filename = fullfile(savedir,sprintf('hdm_dataset%d_class%d_M%d_O%d_Q%d_T%d_ale%d_exp%d_trial%d_iter%d.mat',dataset_id,class,M,O,sum(Q),T-1,useale,experiment,trial,(reload+max_iters)*(useale==1|useale==4)));
if ~exist(filename,'file')
    if useale == 1
        save(filename,'history','loglikelihood','results','params_gen','params_dis','params_ini','RHO_d','RHO_g','N','N_','k_step_D','k_step_G','time','dataset_syn','dataset_g_pos');%,'RMS_d','RMS_g''it'
    elseif useale == 4
        save(filename,'history','loglikelihood','results','params_gen_sets','params_dis','RHO_d','RHO_g','N','N_','k_step_D','k_step_G','time','dataset_syn','dataset_g_pos');%,'RMS_d','RMS_g''it' ,'params_ini_all'
    else
        save(filename,'history','loglikelihood','results','params_ini','params_gen','N','N_','max_iters','time','dataset_syn','dataset_g_pos');
    end
end

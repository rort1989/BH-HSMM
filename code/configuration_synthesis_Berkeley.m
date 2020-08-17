% define parameters for the remaining of the experiment
Qd = 5; % number of state for discriminator
Q = 6*ones(1,10); % use different state for different class, only for generator
M = 1; % number of mixture
num_params = 10;
num_sample = 10;
llh_train = 0; % whether compute the likelihood of training data using learned model
viterbi = 0; % set 1 if using decoded state to generate data, by default, the first training sequence state will be used
useale = 4; % set 0 to use mle learned HMM as generator, 1 to use HDM, 2 to use HDM with fixed hyperparams, 4. Adversarial Bayesian inference
usefilter = 1; % set 1 to use anistropic filtering to post-process sampled data
nfold = 1; % for data synthesis, use all data to train
trial = 1;
savestep = 1; % how often to save checkpoint
N_ = 1000; % number of samples to synthesize

% hyperparameters for learning
reload = 0; % checkpoint: set this to non-zero to load an existing model
max_iters = 2;  % number of epoches
batch_size = 10;
sample_scheme = 1; % 1. sample most possible path; otherwise random path
k_step_D = 1;
k_step_G = 1;
decay_d = 0.9;
decay_g = 0.9;
rmsdecay = 0.9;
epsilon = 1e-6;
alpha = 0.9; % momentum coefficients
clip_thresh = 1; % gradient clipped threshold
clip_val = 0; % gradient clipped value
% if RHO_d is too small, there will be no strong supervision signal on learning
% generator
% RHO_d.dyn = 10^(-4); % -5 step size for dynamic parameters
% RHO_d.mu = 10^(-4); % -3 step size for emission mean parameters
% RHO_d.kappa = 10^(-4); % -4 step size for emission covariance parameters
RHO_d = 5*10^(-2);
% RHO_g.ini = 10^(-3); %-1
% RHO_g.tran = 10^(-3); % -2 % large may flucturate
% RHO_g.mu0 = 10^(-2);  % -1 % large may diverge
% RHO_g.kappa0 = 10^(-2); % -2
% RHO_g.kappa = 10^(-2); % -2
RHO_g = 5*10^(-2);
flag = 0; % convergence flag for early stopping
decay_epoch = 10; % learning rate decay epoches
tol_converge = 10^(-3); % convergence threshold for adversarial learning

% divide into multiple fold, repeat experiment for each fold
% split dataset_scaled according to subjects
cov_prior = 0.1*eye(O);
cov_type = 'diag';
dura_type = 'Poisson';
% select a subset of real data

dataset_name = 'Berkeley';
% class = 1; % Berkeley: 1 jumping, 2 jumping jack, 3 bending, 4 boxing, 8 throwing
% 99 all actions
load(fullfile(dir_base,dataset_name,'feature_orients_clean2.mat'),'feature','labels','Activity_label','nframe_count');
load(fullfile(dir_base,dataset_name,'skel01.mat'),'skel');

topo = [0:6 4 8:13 4 15:20 1 22:27 1 29:34]';
config = 1; % 0: raw data, 1: scale raw data
njoint = 35;
O = 90;
T = 60; % length of data to generate
gap = T; % gap between subsequence
usespeed = 1; % whether to include speed feature
usenorm = 0; % whether to use Taylor's normalization
if class == 99
    class_all = [1:8 10:11];
else
    class_all = class;
end
%~ class dependent info 11 is the total number of possible classes
subjects_selected = cell(11,1);
for c = 1:11
    subjects_selected{c} = 1:12; % all class use same subject set
end
selectallsequence = ones(11,1);

idx_channel_missing = []-3;% []; % set this to non-empty ONLY for experiment 2, whose values will be heldout for the first half of testing sequence
% based on 'idx_joint_to_amc': left leg: 7:13, right arm: 53:62

%% index of sequences from selected subjects
% class_all, subjects_selected, instance
idx_all = [];
subject_all = [];
for c = class_all
    % select all instances
    idx_c = find(labels(:,1) == c);
    if isempty(subjects_selected{c})
        subs = unique(labels(idx_c,2));
    else
        subs = subjects_selected{c};
    end
    for i = 1:length(subs)
        idx_s = find(labels(idx_c,2)==subs(i));
        ii = idx_c(idx_s);
        if selectallsequence(c) == 1 % select all sequences from one subject
            idx_all = union(idx_all,ii);
            subject_all = [subject_all; subs(i)*ones(length(ii),1)];
        else % only select one sequence from one subject
            idx_all = union(idx_all,ii(1));
            subject_all = [subject_all; subs(i)];
        end
    end
end

% construct dataset where each cell contains a complete sequence
N = length(idx_all);
dataset_all = cell(N,1);
for n = 1:N
    dataset_all{n} = feature{idx_all(n)}(:,1:end);    
end
labels = labels(idx_all,:);
nframe_count = nframe_count(idx_all);

%% subsample to get segments of sequences
count = 0;
actions = 0;
subjects = 0;
instances = 0;
dataset = cell(1);
for n = 1:N
    Tn = nframe_count(n);
    if Tn < T % use interpolation
        x = (1:Tn)';
        V = dataset_all{n}';
        samplePoints = {x, 1:O};
        F = griddedInterpolant(samplePoints,V);
        queryPoints = {linspace(1,Tn,T),1:O};
        Vq = F(queryPoints);
        count = count + 1;
        dataset{count,1} = Vq';
        actions(count,1) = labels(n,1);
        subjects(count,1) = labels(n,2);
        instances(count,1) = labels(n,3);
    else % use sliding window
        for t = 1:gap:Tn-T+1
            count = count + 1;
            data = dataset_all{n}(:,t:t+T-1);
            dataset{count,1} = data;
            actions(count,1) = labels(n,1);
            subjects(count,1) = labels(n,2);
            instances(count,1) = labels(n,3);
        end
    end
end
labels = [actions subjects instances];
N = count;

%% normalize to have body-centered orientation of body pose
if usenorm == 1
    Motion = dataset;
    for n = 1:N
        Motion{n} = [zeros(3,size(Motion{n},2)); Motion{n}]';
    end
    % Adopted from Graham Taylor's code
    preprocess1;
    % convert back to the format used for training
    for n = 1:N
        for f = 1:T % stride
            % get joint angle per frame, ignore the offset of body-center
            dataset{n}(:,f) = Motion{n}(f,[1:3 7:O+3])'; % the first three is root joint position, which is set to zeros
        end
    end
end

%% visual inspection
dataset_pos = cell(N,1);
for n = 1:N
    dataset_pos{n} = zeros(njoint*3,T);
    for f = 1:T
        % get joint position per frame
        data = [zeros(1,3) dataset{n}(:,f)']; % the first three is root joint position, which is set to zeros
        XYZ = bvh2xyz_b(skel, data)'; % 3x35        
        dataset_pos{n}(:,f) = XYZ(:);
    end    
    %drawskt3(dataset_pos{n}(1:3:end,:),dataset_pos{n}(3:3:end,:),dataset_pos{n}(2:3:end,:),1:njoint,topo);  %view(-135,30)
    fprintf('%d out of %d is plot. Subject %d\n',n,N,subjects(n));
end

%% scale each dimension to have mean 0 and std 1
if config == 1
    % keep matScale, will be used to rescale data later
    [dataset_scaled, matScale] = scaleData_z(dataset, 0, 1);
    % exclude channels that are mostly constant value i.e. std < threshold
    % left and right clavicle and finger
    idx_channel = find(matScale(:,2) > 0.5);
    [value,idx_channel_missing] = intersect(idx_channel,idx_channel_missing); % only the missing channels that are not excluded due to small variance are kept
    %assert: value = idx_channel(idx_channel_missing)
elseif config == 2 % scale linearly between -180 to 180
    
else
    dataset_scaled = dataset;
    idx_channel = 1:size(dataset_scaled{1},1);
    matScale = ones(length(idx_channel),1);
end
O = length(idx_channel);
for n = 1:N
    dataset_scaled{n} = dataset_scaled{n}(idx_channel,:); % 53 dimensional
end

%% augment positional data with speed feature
if usespeed
    for n = 1:N
        speed = dataset_scaled{n}(:,2:end) - dataset_scaled{n}(:,1:end-1); % double the dimension
        dataset_scaled{n} = [dataset_scaled{n}(:,2:end); speed];
    end
    O = O*2;
    idx_channel_missing = [idx_channel_missing; idx_channel_missing+O/2];
end

%%
clearvars -except savedir filename dataset dataset_scaled dataset_pos T O N config syn class class_all topo skel njoint matScale idx_channel labels nframe_count usespeed subjects subjects_selected idx_channel_missing dataset_id
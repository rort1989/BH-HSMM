% script to quantize the data into discrete visual words
% KD = 20;
rewrite = 1; % set this to 1 if you want to repeat the quantization process
filename = fullfile(savedir,sprintf('dataset%d_class%d_syn%d.mat',dataset_id,class,syn));

% load training data (in position domain)
% load(filename,'T','O','N','matScale','idx_channel','dataset_pos','dataset_scaled','labels');
if rewrite
    clear codebook dataset_quant
else
    load(filename,'codebook','dataset_quant');
end

if ~exist('codebook','var')
    rng default;
    % obtain the code book as the cluster center of K-means
    N = length(dataset_train_r_pos);
    Op = size(dataset_train_r_pos{1},1);
    first_slice_idx = zeros(N+1,1);
    crt_first_slice_idx = 1;
    allsamples = zeros(Op,N*1000);
    for n = 1:N
        first_slice_idx(n) = crt_first_slice_idx;
        crt_first_slice_idx = crt_first_slice_idx + size(dataset_train_r_pos{n},2);
        allsamples(:,first_slice_idx(n):crt_first_slice_idx-1) = dataset_train_r_pos{n};  % D*total_num_slices
    end
    first_slice_idx(N+1) = crt_first_slice_idx;
    allsamples = allsamples(:,1:crt_first_slice_idx-1);
    [IDX,C] = kmeans(allsamples',KD,'Replicates',10);
    codebook.C = C'; % OxT
    codebook.K = KD;
    % obtain the quantized training data
    dataset_quant = cell(N,1);
    for n = 1:N
        dataset_quant{n} = IDX(first_slice_idx(n):first_slice_idx(n+1)-1)'; %%%%% double check if this is row vector
    end
    save(filename,'codebook','dataset_quant','-append');
    fprintf('New quantization K=%d added\n',KD);
end

% quantize the target data using code book
dataset_g_quant = cell(N_,1);
for n = 1:N_
    mat_dist = sqdist(dataset_g_pos{n},codebook.C);
    [~,code] = min(mat_dist,[],2);
    dataset_g_quant{n} = code';
end
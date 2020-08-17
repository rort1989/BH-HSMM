function dataset = augData(dataset,idx_channel,matScale)

N = length(dataset);
idx_ = setdiff(1:size(matScale,1),idx_channel); % channel that is not selected
for n = 1:N
    T = size(dataset{n},2);
    data = dataset{n};
    dataset{n} = zeros(size(matScale,1),T);
    dataset{n}(idx_channel,:) = data;
    dataset{n}(idx_,:) = repmat(matScale(idx_,1),1,T); % now obs_scaled{n} is 62xT
end
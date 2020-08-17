function dataset_pos = angle2pos(dataset_angle,skel,njoint)

N_ = length(dataset_angle);
dataset_pos = cell(N_,1); % synthetic pose
for n = 1:N_
    T_ = size(dataset_angle{n},2);
    dataset_pos{n} = zeros(njoint*3,T_);
    for f = 1:T_
        % get joint position per frame for synthetic data
        data = dataset_angle{n}(:,f)'; % the first three is root joint position, which is set to zeros
        XYZ = acclaim2xyz(skel, data);
        temp = XYZ'; % 31x3
        dataset_pos{n}(:,f) = temp(:);
    end
end
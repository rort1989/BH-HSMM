function output = anisotropic_diffusion_filter(input,lambda,k,max_iter)

N = size(input,2);
I = [input; 1:N];
Ir = I;
        
% anisotropic diffusion iteration
for iter = 1:max_iter
    I = Ir;    
    for j = 2:N-1
        NI = I(:,j-1) - I(:,j);
        SI = I(:,j+1) - I(:,j);
        Ir(:,j) = I(:,j) + lambda * (exp( - NI'*NI/k/k) * NI + exp( - SI'*SI/k/k) * SI);
    end
end

output = I(1:end-1,:);
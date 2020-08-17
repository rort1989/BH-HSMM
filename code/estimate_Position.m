function y = estimate_Position(params, state)
% Reference: Matthew Brand, Voice Puppetry, 1999

T = length(state);
O = size(params.mu,1)/2;
Q = size(params.mu,2);
K = zeros(2*O,2*O,Q);
for q = 1:Q
    K(:,:,q) = inv(params.sigma(:,:,q));
end
% solve Ay = b
% construct A
A = zeros(O*T,O*T);
b = zeros(O*T,1);
% t = 1
q_ = state(2);
Kxx_ = K(1:O,1:O,q_);
Kxs_ = K(1:O,1+O:2*O,q_);
Ksx_ = K(1+O:2*O,1:O,q_);
Kss_ = K(1+O:2*O,1+O:2*O,q_);
A(1:O,1:2*O) = [Kss_  -Kss_-0.5*(Kxs_+Ksx_)];
b(1:O) = - 0.5*(Kxs_+Ksx_)*params.mu(1:O,q_) - Kss_*params.mu(1+O:2*O,q_);
for t = 2:T-1
    q = state(t);
    Kxx = K(1:O,1:O,q);
    Kxs = K(1:O,1+O:2*O,q);
    Ksx = K(1+O:2*O,1:O,q);
    Kss = K(1+O:2*O,1+O:2*O,q);    
    q_ = state(t+1);
    Kxx_ = K(1:O,1:O,q_);
    Kxs_ = K(1:O,1+O:2*O,q_);
    Ksx_ = K(1+O:2*O,1:O,q_);
    Kss_ = K(1+O:2*O,1+O:2*O,q_);
    
    A((t-1)*O+1:t*O,(t-2)*O+1:(t+1)*O) = [-Kss-0.5*(Kxs+Ksx)  Kxx+Kss+Kxs+Ksx+Kss_  -Kss_-0.5*(Kxs_+Ksx_)]; % 3*O
    b((t-1)*O+1:t*O) = (Kxx+0.5*(Kxs+Ksx))*params.mu(1:O,q) + (Kss+0.5*(Kxs+Ksx))*params.mu(1+O:2*O,q) - 0.5*(Kxs_+Ksx_)*params.mu(1:O,q_) - Kss_*params.mu(1+O:2*O,q_);
end
% t = T
q = state(T);
Kxx = K(1:O,1:O,q);
Kxs = K(1:O,1+O:2*O,q);
Ksx = K(1+O:2*O,1:O,q);
Kss = K(1+O:2*O,1+O:2*O,q);
A((T-1)*O+1:T*O,(T-2)*O+1:T*O) = [-Kss-0.5*(Kxs+Ksx) Kxx+Kss+Kxs+Ksx];
b((T-1)*O+1:T*O) = (Kxx+0.5*(Kxs+Ksx))*params.mu(1:O,q) + (Kss+0.5*(Kxs+Ksx))*params.mu(1+O:2*O,q);

y = A\b;
y = reshape(y,O,T);
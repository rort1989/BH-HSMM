function GRAD = grad_clip(GRAD,clip,val)
% clip the gradient: set to 0 if GRAD is NaN or too large/small

for n = 1:length(GRAD)
    g = GRAD{n};
    g(isnan(g)) = val;
    g(abs(g)>clip) = val;
    GRAD{n} = g;
end
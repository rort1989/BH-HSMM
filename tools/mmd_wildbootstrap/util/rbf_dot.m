%Radial basis function inner product
%Arthur Gretton

%Pattern input format : [pattern1 ; pattern2 ; ...]
%Output : p11*p21 p11*p22 ... ; p12*p21 ...

function kernel=rbf_dot(patterns)
    sigma = median_heur([patterns]);
    if isnan(sigma)
        sigma = 1;
        warning('median heuristic failed')
    end
    kernel = @(X,Y) rbf_dot_deg(X,Y,sigma);
end


function [H]=rbf_dot_deg(X,Y,deg)

size1=size(X);
size2=size(Y);

G = sum((X.*X),2);
H = sum((Y.*Y),2);

Q = repmat(G,1,size2(1));
R = repmat(H',size1(1),1);

H = Q + R - 2*X*Y';


H=exp(-H/2/deg^2);

end
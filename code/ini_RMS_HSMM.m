function [RMS_d,RMS_g] = ini_RMS_HSMM(Q,O,Qd)
% generator: HSMM, discriminator: HSMM, duration: Poisson
if nargin < 3
    Qd = Q;
end

% GRAD_pi = zeros(Q,1);
% GRAD_A = zeros(Q,Q);
% GRAD_kappa_C = zeros(Q,1);
% GRAD_mu = zeros(O,Q);
% GRAD_kappa = zeros(O,Q);
RMS_d = cell(5,1);
RMS_d{1} = zeros(Qd,1);
RMS_d{2} = zeros(Qd,Qd);
RMS_d{3} = zeros(Qd,1);
RMS_d{4} = zeros(O,Qd);
RMS_d{5} = zeros(O,Qd);

% GRAD_pi = zeros(Q,1);
% GRAD_A = zeros(Q,Q);
% GRAD_kappa_C = zeros(Q,1);
% GRAD_mu = zeros(O,Q);
% GRAD_kappa = zeros(O,Q);
RMS_g = cell(5,1);
RMS_g{1} = zeros(Q,1);
RMS_g{2} = zeros(Q,Q);
RMS_g{3} = zeros(Q,1);
RMS_g{4} = zeros(O,Q);
RMS_g{5} = zeros(O,Q);
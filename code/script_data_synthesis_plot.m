% plot results
if useale == 1
%     count = numel(history.loss_d);
% %     % generator likelihood    
% %     figure;
% %     plot(1:count,history.llh_p(1:count),1:count,history.llh_n(1:count),'k'); %1:count,history.llh_g(1:count),'g',   mean(loglikelihood)*ones(1,count)  ,1:count,history.llh_r(1:count),'r'
% %     title('loglikelihood')
% %     legend({'logP(x_g|\phi^+)','logP(x_g|\phi^-)'},'Location','southeast') % 'logP(x_g|\theta_g)', ,'logP(x_g|\alpha_r)'
% %     xlabel('iteration')
% %     % discriminator probability
% %     figure;
% %     plot(1:count,history.prob_r_dis(1:count),'r',1:count,history.prob_g_dis(1:count))
% %     title('P(y=1|x,\phi)')
% %     legend({'P(y=1|x^+,\phi)','P(y=1|x^-,\phi)'},'Location','southeast')
% %     xlabel('iteration')
    figure;
    subplot(2,1,1)
%     plot(1:count,history.loss_d(1:count))%per batch, noisy
    plot(mean(history.loss_d,1)); % per epoch, averaged over all batches
    title('loss(D)')
    %xlabel('iteration')
    xlabel('epoch')
    subplot(2,1,2)
%     plot(1:count,history.loss_g(1:count))
    plot(mean(history.loss_g,1)); %
    title('loss(G)')
    %xlabel('iteration')
    xlabel('epoch')
end
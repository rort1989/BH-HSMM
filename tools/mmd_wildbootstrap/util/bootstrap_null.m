function [ results ] = bootstrap_null(m,numBootstrap,...
    statMatrix,alpha,wild_bootstrap,test_type)

processes = wild_bootstrap(m,numBootstrap);

testStat = m*mean2(statMatrix);

testStats = zeros(numBootstrap,1);
for process = 1:numBootstrap
    mn = mean(processes(:,process));
    if test_type==1
        matFix = (processes(:,process)-mn)*(processes(:,process)-mn)';
    else
        matFix = processes(:,process)*processes(:,process)';
    end
    testStats(process) =  m*mean2(statMatrix.*matFix );
end

results.testStat = testStat;
results.quantile = quantile(testStats,1-alpha);
results.reject = testStat > results.quantile;
end


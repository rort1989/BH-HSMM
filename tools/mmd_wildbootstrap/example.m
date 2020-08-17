% addpath('util')
n = 1000;
rng(2)

if exist('arima')
    model = arima('Constant',0,'AR',{0.5},'Variance',1);
    processes = simulate(model,n,'numPaths',2);
else
    w=randn(n,2);
    a = [1 -0.5];
    processes=filter(1,a,w);
end
 
 X = processes(:,1);
 Y = processes(:,2);

%these are independent - test should accpet i.e. reject=0 (keep in mind that the 
%null hypothesis rejection rate is 5% )
disp('reject should be 0, test Stat should be smaller than 95% quantile')
disp(wildHSIC(X,Y))

%same function with non-default parameters and proceess with different
%distibutions. 
disp('reject should be 1, test Stat should be larger than 99% quantile')
disp(wildHSIC(X,X+Y,'TestType',1,'Alpha',0.01,'NumBootstrap',500))


disp('=== Now MMD ===')


%these are the same. 
disp('reject should be 0, test Stat should be smaller than 95% quantile')
disp(wildMMD(X,Y))

%same function with non-default parameters and proceess with different
%distibutions
disp('reject should be 1, test Stat should be larger than 99% quantile')
disp(wildMMD(X,2*Y,'TestType',2,'Alpha',0.01,'NumBootstrap',500))


%mulitdimensional case

if exist('arima')
    processes = simulate(model,n,'numPaths',8);
else
    w=randn(n,8);
    processes=filter(1,a,w);
end
X = processes(:,1:4);
Y = processes(:,5:8);

disp('reject should be 1, test Stat should be greater than 95% quantile')
disp(wildHSIC(X,Y+X));


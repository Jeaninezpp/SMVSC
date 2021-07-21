clear;
clc;
warning off;

addpath(genpath('./'));

ds{1} = "Caltech101-20";        % 2386

dsPath = './';

lambda=1;

dataName = ds{1}; disp(dataName);
load(strcat(dsPath,dataName));


k = length(unique(Y));
anchor = [k 2*k 3*k];
d = k;

for ia = 1:length(anchor)
    tic;
    [U,V,A,W,Z,iter,obj] = algo_qp(X,Y,lambda,d,anchor(ia)); % X,Y,lambda,d,numanchor
    res = myNMIACCwithmean(U,Y,k);
    timer  = toc;
    fprintf('Anchor:\t%d \t ACC:%12.6f \t NMI:%12.6f \t Purity:%12.6f \t Fscore:%12.6f \t Time:%12.6f \n',[anchor(ia) res(1) res(2) res(3) res(4) timer]);
end



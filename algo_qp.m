function [UU,V,A,W,Z,iter,obj] = algo_qp(X,Y,lambda,d,numanchor)
% m      : the number of anchor. the size of Z is m*n.
% lambda : the hyper-parameter of regularization term.
% X      : n*di

%% initialize
maxIter = 50 ; % the number of iterations

m = numanchor;
numclass = length(unique(Y));
numview = length(X);
numsample = size(Y,1);

W = cell(numview,1);            % di * d
A = zeros(d,m);         % d  * m
Z = zeros(m,numsample); % m  * n

for i = 1:numview
   X{i} = mapstd(X{i}',0,1); % turn into d*n
   di = size(X{i},1); 
   W{i} = zeros(di,d);
end
Z(:,1:m) = eye(m);


alpha = ones(1,numview)/numview;
opt.disp = 0;

flag = 1;
iter = 0;
%%
while flag
    iter = iter + 1;
    
    %% optimize W_i
    AZ = A*Z; % since each view share the same A and Z.
    parfor iv=1:numview
        C = X{iv}*AZ';      
        [U,~,V] = svd(C,'econ');
        W{iv} = U*V';
    end

    %% optimize A
    sumAlpha = 0;
    part1 = 0;
    for ia = 1:numview
        al2 = alpha(ia)^2;
        sumAlpha = sumAlpha + al2;
        part1 = part1 + al2 * W{ia}' * X{ia} * Z';
    end
    [Unew,~,Vnew] = svd(part1,'econ');
%     A = (part1/sumAlpha) * inv(Z*Z');
    A = Unew*Vnew';
    % Replace inv(A)*b with A\b
    % Replace b*inv(A) with b/A
    
    %% optimize Z
    % % QP
% Sbar=[];
% H = 2*sumAlpha*A'*A+2*lambda*eye(m);
H = 2*sumAlpha*eye(m)+2*lambda*eye(m);
H = (H+H')/2;
% [r,q] = chol(H);

options = optimset( 'Algorithm','interior-point-convex','Display','off'); % Algorithm 默认为 interior-point-convex
parfor ji=1:numsample
    ff=0;
    for j=1:numview
        C = W{j} * A;
        ff = ff - 2*X{j}(:,ji)'*C;
    end
    Z(:,ji) = quadprog(H,ff',[],[],ones(1,m),1,zeros(m,1),ones(m,1),[],options);
end

    %% optimize alpha
    M = zeros(numview,1);
    for iv = 1:numview
        M(iv) = norm( X{iv} - W{iv} * A * Z,'fro')^2;
    end
    Mfra = M.^-1;
    Q = 1/sum(Mfra);
    alpha = Q*Mfra;

    %%
    term1 = 0;
    for iv = 1:numview
        term1 = term1 + alpha(iv)^2 * norm(X{iv} - W{iv} * A * Z,'fro')^2;
    end
    term2 = lambda * norm(Z,'fro')^2;
    obj(iter) = term1+ term2;
    
    
    if (iter>9) && (abs((obj(iter-1)-obj(iter))/(obj(iter-1)))<1e-3 || iter>maxIter || obj(iter) < 1e-10)
        [UU,~,V]=svd(Z','econ');
        flag = 0;
    end
end
         
         
    

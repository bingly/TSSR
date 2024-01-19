%% Load Setting
clear;
addpath([pwd, '/funs']);
addpath([pwd, '/Nuclear_norm_l21_Algorithm']);
addpath([pwd, '/ClusteringMeasure']);
addpath(genpath('proximal_operator'));
addpath(genpath('tSVD'));
addpath(genpath('utils'));
addpath(genpath('Ncut_9'));

dataname='reutersMulSubset';  %%ORL MSRC_v1 EYaleB10 COIL20 BBCSport reutersMulSubset Caltech101_20 scene15 UCI Hdigit yale
load(strcat('../data/',dataname,'.mat'));

alpha=.01;beta=.001;scalar=.5;
% alpha = .1;  % ORL(0.2)  ÆäËû£¨0.01£© start: COIL20, lambda1=0.0001,lambda2=10.0000,lambda3=0.0010:
% beta = .1;
% scalar = 1;
DEBUG = 0;
fprintf('start: %s, alpha=%.4f,beta=%.4f,s=%.4f:\n',dataname,alpha,beta,scalar);

%% preparation
cls_num = length(unique(gt));
K = length(X); N = size(X{1},2); %sample number
sX = [N, N, K];

%% Optimizataion
tic;
epson = 1e-7; max_mu = 10e10; pho_mu = 2;
max_iter = 50;

params = [.0001, .001, .01, .1, 1, 10, 100, 1000];
scales = [.1, .5, 1, 1.5, 2, 5, 10];


mu = 10e-5;
iter = 0;

%% Initialize and Settings
I = eye(N); %identity tensor
for k=1:K
    Xtx{k} = X{k}'*X{k};
    Z{k} = zeros(N,N);  %represent C
end

J = zeros(N,N,K);
Y=J;

w = 1/K*ones(1,K);  %view weighted 
s = zeros(K,1);

while iter < max_iter
    %% Update Z^i: K
    for k=1:K
        %derivation
        Q = pinv(alpha*w(k)*Xtx{k}+(mu/2)*I) * (alpha*w(k)*Xtx{k}+mu*(J(:,:,k)-Y(:,:,k)/mu)) / scalar;
        %projection
        if(strcmp(dataname, 'COIL20'))
           Q = double(Q); 
        end
        Q = SimplexProj(Q');
        Z{k} = scalar*Q';
    end

    %% update J
    Zt = cat(3, Z{:,:});
    z = Zt(:);
    y = Y(:);
    [j, ~] = wshrinkObj(z + 1/mu*y,1/mu,sX,0,3);
    J = reshape(j, sX);

    %% update w
    for k = 1:K
        s(k) = norm(X{k}-X{k}*Z{k} ,'fro')^2;
    end

    qH = eye(K);
    qf = alpha/beta * s;
    Aeq=ones(1,K);
    beq=1;    
    LB=zeros(K,1);
    UB=ones(K,1);
    opts = optimset('Display','off');
    w = quadprog(qH,qf,[],[],Aeq,beq,LB,UB,[],opts);

    %% check convergence
    leq = Zt-J;
    leqm = max(abs(leq(:)));

    err = max(leqm);
    if DEBUG
        fprintf('iter = %2d, %4.8f \n',iter,leqm);
    end
    if err < epson
        break;
    end

   %% update Lagrange multiplier and  penalty parameter beta
    Y = Y + mu*leq;

    mu = min(mu*pho_mu, max_mu);
    iter = iter + 1;
end

Aff = 0;
for k=1:K
    Aff = Aff +  w(k)*(abs(Z{k})+abs(Z{k}'));
end

if(strcmp(dataname, 'COIL20'))
   Aff = double(Aff); 
end
% 
% figure(1); imagesc(Aff);
% 
% figure(2);
% Y = tsne(Aff,'Algorithm','exact','Distance','seuclidean'); 
% gscatter(Y(:,1), Y(:,2),gt);

for index = 1:10
    clu = SpectralClustering(Aff, cls_num);

    [A nmi(index) avgent] = compute_nmi(gt,clu);
    ACC(index) = Accuracy(clu,double(gt));
    [f(index),p(index),r(index)] = compute_f(gt,clu);
    [AR(index),RI,MI,HI]=RandIndex(gt,clu);
    fprintf('%d: %.4f %.4f %.4f %.4f %.4f %.4f\n',index,ACC(index),nmi(index),AR(index),f(index),p(index),r(index));
    toc;
end
fprintf('TSSR,%.3f¡À%.3f,%.3f¡À%.3f,%.3f¡À%.3f,%.3f¡À%.3f,%.3f¡À%.3f,%.3f¡À%.3f \n',...
    mean(ACC),std(ACC),mean(nmi),std(nmi),mean(AR),std(AR),mean(f),std(f),mean(p),std(p),mean(r),std(r));

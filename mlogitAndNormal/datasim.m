% Simple simulation of mlogit and wage regression with unobserved heterogeneity
% Types are treated as observed in this case

clear all; clc;
addpath('../GeneralFunctions')
tic;
seed = 1234;
rng(seed,'twister');
 
N       = 1e5;
T       = 5;
J       = 5;

type  = rand(N,1)>.3;
typew = repmat(type,[1 T]);

% generate the choice data
X = [ones(N*T,1) 5+3*randn(N*T,1) rand(N*T,1) 2.5+2*randn(N*T,1) typew(:)];
for j=1:J
    Z(:,:,j) = [3+randn(N*T,1) randn(N*T,1)-1 rand(N*T,1)];
end

% X coefficients
b(:,1) = [-0.15; 0.10; 0.50; 0.10; -.15];
b(:,2) = [-1.50; 0.15; 0.70; 0.20;  .25];
b(:,3) = [-0.75; 0.25;-0.40; 0.30; -.05];
b(:,4) = [ 0.65; 0.05;-0.30; 0.40;  .35];
b(:,5) = [ 0.75; 0.10;-0.50; 0.50; -.25];

% Z coefficients
bz = [.2;.5;.8];

% lnWage coefficients
bwAns(:,1) = [-0.15; 0.10; 0.50; 0.10; -.15];
sigwAns    = .3;

% generate choice probabilities
dem = zeros(N*T,1);
for j=1:J
    u(:,j) = X*b(:,j)+Z(:,:,j)*bz;
    dem=exp(u(:,j))+dem;
end
for j=1:J
    p(:,j) = exp(u(:,j))./dem;
end

% use the choice probabilities to create the observed choices
draw=rand(N*T,1);
Y=(draw<sum(p(:,1:end),2))+...
  (draw<sum(p(:,2:end),2))+...
  (draw<sum(p(:,3:end),2))+...
  (draw<sum(p(:,4:end),2))+...
  (draw<sum(p(:,5:end),2));
tabulate(Y);

bAns = b(:)-repmat(b(:,J),J,1);
bAns = cat(1,bAns(1:(J-1)*size(X,2)),bz);
size(bAns)

% generate wages
Xwage  = X;
lnWage = X*bwAns+sigwAns*randn(N*T,1);

disp(['Time spent on simulation: ',num2str(toc)]);

% Now estimate choice parameters as if unobserved type is observed:
options=optimset('Disp','Iter','LargeScale','on','MaxFunEvals',2000000,'MaxIter',15000,'TolX',1e-8,'Tolfun',1e-8,'GradObj','on','DerivativeCheck','off','FinDiffType','central');
startval = rand(size(bAns));
[bEst,lEst,~,~,~,hEst] = fminunc('clogit',startval,options,[],Y,X,Z);
hEst = full(hEst);
se = sqrt(diag(inv(hEst)));
[bEst bAns]
[bEst se]

% Now estimate wage parameters as if unobserved type is observed:
startval = rand(length(bwAns)+1,1);
[bwEst,lwEst,~,~,~,hwEst] = fminunc('normalMLE',startval,options,[],lnWage,Xwage,[]);
hwEst = full(hwEst);
se = sqrt(diag(inv(hwEst)));
[bwEst cat(1,bwAns,sigwAns)]
[bwEst se]

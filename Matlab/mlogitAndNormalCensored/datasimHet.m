% Simple simulation of mlogit

clear all; clc;
addpath('../GeneralFunctions')
tic;
seed = 1234;
rng(seed,'twister');
 
N       = 1e5;
T       = 5;
S       = 2;
J       = 5;
piAns   = .3;

IDw   = [1:N]'*ones(1,T);
IDl   = cat(1,IDw(:),IDw(:));
type  = rand(N,1)>piAns;
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
sigWans    = .3;

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
lnWage = Xwage*bwAns+sigWans*randn(N*T,1);
wageflag = ismember(Y,[2 3 4]);
lnWage(~wageflag) = -999;

disp(['Time spent on simulation: ',num2str(toc)]);

% Now estimate without knowing unobserved types:
Xfeas        = cat(2,kron(ones(S,1),X(:,1:end-1)),kron(eye(S,S-1),ones(length(X),1)));
Xwagefeas    = cat(2,kron(ones(S,1),Xwage(:,1:end-1)),kron(eye(S,S-1),ones(length(Xwage),1)));
wageflagfeas = kron(ones(S,1),wageflag);
Yfeas        = kron(ones(S,1),Y);
lnWfeas      = kron(ones(S,1),lnWage);
Zfeas = [];
for s=1:S
	Zfeas = cat(1,Zfeas,Z);
end
options=optimset('Disp','Iter','LargeScale','on','MaxFunEvals',2000000,'MaxIter',15000,'TolX',1e-8,'Tolfun',1e-8,'GradObj','on','DerivativeCheck','off','FinDiffType','central');

% EM algorithm starting values
prior = [.55 .45];
bEst = bAns + .5*rand(size(bAns)).*bAns - .25*bAns;
bwEst = [bwAns;sigWans] + .5*rand(length(bwAns)+1,1).*[bwAns;sigWans] - .25*[bwAns;sigWans];
EMcrit = 1;
iteration = 1;

full_like = likecalc(Yfeas,lnWfeas,Xfeas,Xwagefeas,Zfeas,bEst,bwEst,wageflagfeas,N,T,S,J);
[prior,Ptype,Ptypel,jointlike] = typeprob(prior,full_like,T);
disp(['Initial likelihood value = ',num2str(jointlike)]);

while EMcrit>1e-5
	oPtype = Ptype;
	full_like = likecalc(Yfeas,lnWfeas,Xfeas,Xwagefeas,Zfeas,bEst,bwEst,wageflagfeas,N,T,S,J);
	[prior,Ptype,Ptypel,jointlike] = typeprob(prior,full_like,T);
	[bEst]  = fminunc('clogit'   ,bEst ,options,[],Yfeas  ,Xfeas,Zfeas,[],Ptypel);
	[bwEst] = fminunc('normalMLE',bwEst,options,[],lnWfeas(wageflagfeas==1),Xwagefeas(wageflagfeas==1,:),Ptypel(wageflagfeas==1));
	EMcrit = norm(Ptype(:)-oPtype(:),Inf);
	iteration = iteration+1;
	disp(['Likelihood value = ',num2str(jointlike)]);
	disp(['Pr(type==1) is ',num2str(prior(1))]);
	disp(['Iteration is ',num2str(iteration)]);
	disp(['EM criterion is ',num2str(EMcrit)]);
end

% re-estimate to get choice model hessian for statistical inference
[bEst,lEst,~,~,~,hEst] = fminunc('clogit',bEst,options,[],Yfeas,Xfeas,Zfeas,[],Ptypel);
hEst = full(hEst);
se = sqrt(diag(inv(hEst)));
[bEst bAns]
[bEst se]

% re-estimate to get wage model hessian for statistical inference
[bwEst,lwEst,~,~,~,hwEst] = fminunc('normalMLE',bwEst,options,[],lnWfeas(wageflagfeas==1),Xwagefeas(wageflagfeas==1,:),Ptypel(wageflagfeas==1));
hwEst = full(hwEst);
se = sqrt(diag(inv(hwEst)));
[bwEst cat(1,bwAns,sigWans)]
[bwEst se]

[prior(1) piAns mean(type)]
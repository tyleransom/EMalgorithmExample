% Simple simulation of mlogit

clear all; clc;
addpath('../GeneralFunctions')
tic;
seed = 1234;
rng(seed,'twister');
 
N       = 1e5;
T       = 5;
S       = 2;
J       = 6;
piAns   = .3;

IDw   = [1:N]'*ones(1,T);
IDl   = cat(1,IDw(:),IDw(:));
type  = rand(N,1)>piAns;
typew = repmat(type,[1 T]);

% generate the choice X's and Z's
X = [ones(N*T,1) 5+3*randn(N*T,1) rand(N*T,1) 2.5+2*randn(N*T,1) typew(:)];
for j=1:J
    Z(:,:,j) = [3+randn(N*T,1) randn(N*T,1)-1 rand(N*T,1)];
end

% generate the wage X's
X1wage = [X(:,1:end-1) rand(N*T,1) 2.5+2*randn(N*T,1) typew(:)];

% generate the employment X's
X1emp = [X(:,1:end-1) 2.5+2*randn(N*T,1) rand(N*T,1) 2.5+2*randn(N*T,1) typew(:)];

% X coefficients
b(:,1) = [ 1.15; 0.10; 0.50; 0.10; -.15];
b(:,2) = [ 0.90; 0.15; 0.70; 0.20;  .25];
b(:,3) = [ 0.75; 0.25;-0.40; 0.30; -.05];
b(:,4) = [ 0.65; 0.05;-0.30; 0.40;  .35];
b(:,5) = [ 0.75; 0.10;-0.50; 0.50; -.25];
b(:,6) = [-0.65; 0.05; 0.30; 0.40;  .15];

% Z coefficients
bz = [.2;.5;.8];

% lnWage coefficients
bwAns(:,1) = rand(size(X1wage,2),1)-.5;
bwAns = round(bwAns,1)
sigWans    = .3;

% employment coefficients
beAns(:,1) = .5*rand(size(X1emp,2),1)-.25;
beAns(1) = beAns(1)-2;
beAns = round(beAns,1)

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
Y = zeros(N*T,1);
for j=1:J
	Ytemp = (draw<sum(p(:,j:end),2));
	Y = Y+Ytemp;
end
tabulate(Y);

bAns = b(:)-repmat(b(:,J),J,1);
bAns = cat(1,bAns(1:(J-1)*size(X,2)),bz);
size(bAns)

% generate employment
Xemp = X1emp;
for j=1:J
    pe = exp(Xemp*beAns)./(1+exp(Xemp*beAns));
end
emp = rand(N*T,1)<pe;
empflag = Y<(J/2+1);
emp(~empflag) = -999;
tabulate(emp);
tabulate(emp(Y<(J/2+1)));

% generate wages
Xwage  = X1wage;
lnWage = Xwage*bwAns+sigWans*randn(N*T,1);
wageflag = Y<(J/2+1) & emp==1;
lnWage(~wageflag) = -999;

disp(['Time spent on simulation: ',num2str(toc)]);
options=optimset('Disp','Iter','LargeScale','on','MaxFunEvals',2000000,'MaxIter',15000,'TolX',1e-8,'Tolfun',1e-8,'GradObj','on','DerivativeCheck','off','FinDiffType','central');

% re-estimate to get choice model hessian for statistical inference
[bEst,lEst,~,~,~,hEst] = fminunc('clogit',rand(size(bAns)),options,[],Y,X,Z,[]);
hEst = full(hEst);
se = sqrt(diag(inv(hEst)));
[bEst bAns]
[bEst se]

% re-estimate to get employment model hessian for statistical inference
[beEst,~,bestats] = glmfit(Xemp(empflag==1,2:end),emp(empflag==1),'binomial','link','logit');
[beEst beAns]
[beEst bestats.se]

% re-estimate to get wage model hessian for statistical inference
[bwEst,lwEst,~,~,~,hwEst] = fminunc('normalMLE',rand(size(cat(1,bwAns,sigWans))),options,[],lnWage(wageflag==1),Xwage(wageflag==1,:),[],[]);
hwEst = full(hwEst);
se = sqrt(diag(inv(hwEst)));
[bwEst cat(1,bwAns,sigWans)]
[bwEst se]

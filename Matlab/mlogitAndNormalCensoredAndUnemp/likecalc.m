function full_like = likecalc(Y,emp,lnWage,X,Xemp,Xwage,Z,b,be,bwfull,empflag,wageflag,N,T,S,J)
bw   = bwfull(1:end-1);
sigw = bwfull(end);

Pe2 = glmval(be,Xemp(:,2:end),'logit');
Pe1 = pclogit(be,emp+1,Xemp,[],1);
Pe = exp(Xemp*be)./(1+exp(Xemp*be));
P  = pclogit(b,Y,X,Z);
assert(norm(Pe-Pe1(:,end))<1e-12,'P''s not computed in same way!')
assert(norm(Pe-Pe2(:,end))<1e-12,'P''s not computed in same way!')

Ywtemp = reshape(Y,[N T S]);
Pw = zeros(N,T,S,J);
Yw = zeros(N,T,S,J);
for j=1:J
	Pw(:,:,:,j) = reshape(P(:,j),[N T S 1]);
	Yw(:,:,:,j) = Ywtemp==j;
end

choice_like = ones(N,S);
for s=1:S
	for j=1:J
		choice_like(:,s) = squeeze(prod(prod(Pw(:,:,s,:).^Yw(:,:,s,:),4),2));
	end
end

wageflagw = reshape(wageflag,[N T S]);
lnWagew   = reshape(lnWage,[N T S]);
wageResw  = reshape(Xwage*bw,[N T S]);
wage_like = ones(N,S);
for s=1:S
	wage_like(:,s) = squeeze(prod(normpdf(lnWagew(:,:,s)-wageResw(:,:,s),0,sigw).^(wageflagw(:,:,s)==1),2));
end

empflagw = reshape(empflag,[N T S]);
empw     = reshape(emp,[N T S]);
Pew      = reshape(Pe(:,end),[N T S]);
emp_like = ones(N,S);
for s=1:S
	emp_like(:,s) = squeeze(prod((Pew(:,:,s).^(empw(:,:,s)==1).*(1-Pew(:,:,s)).^(1-(empw(:,:,s)==1))).^(empflagw(:,:,s)==1),2));
end

full_like = choice_like.*wage_like.*emp_like;
end
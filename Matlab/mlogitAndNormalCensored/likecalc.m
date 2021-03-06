function full_like = likecalc(Y,lnWage,X,Xwage,Z,b,bwfull,wageflag,N,T,S,J)
bw   = bwfull(1:end-1);
sigw = bwfull(end);

P = pclogit(b,Y,X,Z);

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

full_like = choice_like.*wage_like;
end
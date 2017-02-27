function full_like = likecalc(Y,lnWage,X,Xwage,Z,b,bwfull,wageflag,N,T,S,J)
bw   = bwfull(1:end-1);
sigw = bwfull(end);

P = pclogit(b,Y,X,Z);

wageflagw = permute(reshape(wageflag,[N S T]),[1 3 2]);
lnWagew   = permute(reshape(lnWage,[N S T]),[1 3 2]);
Yw        = permute(reshape(Y,[N S T]),[1 3 2]);
wageResw  = permute(reshape(Xwage*bw,[N S T]),[1 3 2]);
Pw        = zeros(N,T,S,J);
for j=1:J
	Pw(:,:,:,j) = permute(reshape(P(:,j),[N S T 1]),[1 3 2 4]);
end

choice_like = ones(N,S);
for s=1:S
	for j=1:J
		choice_like(:,s) = choice_like(:,s).*squeeze(prod((Pw(:,:,s,j).^(Yw(:,:,s)==j)),2));
	end
end

wage_like = ones(N,S);
for s=1:S
	wage_like(:,s) = squeeze(prod(normpdf(lnWagew(:,:,s)-wageResw(:,:,s),0,sigw).^(wageflagw(:,:,s)==1),2));
end

full_like = choice_like.*wage_like;
end
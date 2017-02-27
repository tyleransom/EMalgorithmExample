function full_like = likecalc(Y,X,Z,b,N,T,S,J)

P = pclogit(b,Y,X,Z);

Yw = permute(reshape(Y,[N S T]),[1 3 2]);
Pw = zeros(N,T,S,J);
for j=1:J
	Pw(:,:,:,j) = permute(reshape(P(:,j),[N S T 1]),[1 3 2 4]);
end

choice_like = ones(N,S);
for s=1:S
	for j=1:J
		choice_like(:,s) = choice_like(:,s).*squeeze(prod((Pw(:,:,s,j).^(Yw(:,:,s)==j)),2));
	end
end

full_like = choice_like;
end
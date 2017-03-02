function full_like = likecalc(Y,X,Z,b,N,T,S,J)

P = pclogit(b,Y,X,Z);

% Yw = permute(reshape(Y,[N S T]),[1 3 2]);
Ywtemp = reshape(Y,[N T S]);
Pw = zeros(N,T,S,J);
Yw = zeros(N,T,S,J);
for j=1:J
	% Pw(:,:,:,j) = permute(reshape(P(:,j),[N S T 1]),[1 3 2 4]);
	Pw(:,:,:,j) = reshape(P(:,j),[N T S 1]);
	Yw(:,:,:,j) = Ywtemp==j;
end

choice_like = ones(N,S);
for s=1:S
	for j=1:J
		choice_like(:,s) = squeeze(prod(prod(Pw(:,:,s,:).^Yw(:,:,s,:),4),2));
	end
end

full_like = choice_like;
end
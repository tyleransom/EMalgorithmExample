function [prior,Ptype,Ptypel,jointlike] = typeprob(prior,base,T)

N = size(base,1);
S = size(base,2);

for s=1:S
	Ptype(:,s) = prior(s)*base(:,s)./(base*prior');
end

prior = mean(Ptype);
jointlike = sum(log(base*prior'));

Ptypew = zeros(N,T,S);
for s=1:S
	Ptypew(:,:,s) = repmat(Ptype(:,s),[1 T 1]);
end
Ptypel = reshape(Ptypew,[N*T*S 1]);
end
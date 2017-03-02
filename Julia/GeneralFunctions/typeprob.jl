function typeprob(prior,full_like,T)

N = size(full_like,1)
S = size(full_like,2)
T = convert(Int64,T)

Ptype = zeros(N,S)
for s=1:S
    Ptype[:,s] = prior[s]*full_like[:,s]./(full_like*prior')
end
@assert norm(sum(Ptype,2)-ones(N),2)<1e-10

prior = mean(Ptype,1)
jointlike = sum(log(full_like*prior'))

Ptypew = zeros(N,T,S)
for s=1:S
    Ptypew[:,:,s] = repmat(Ptype[:,s],1,T)
end
Ptypel = reshape(Ptypew,N*T*S,1)

return prior,Ptype,Ptypel,jointlike
end
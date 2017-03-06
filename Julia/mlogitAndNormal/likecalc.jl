function likecalc(Y,X,Z,lnWage,Xwage,ßest,ωest,N,T,S,J)
ω = ωest[1:end-1]
σ = ωest[end]

thisNormal = Normal(0,σ)

N = convert(Int64,N)
T = convert(Int64,T)
S = convert(Int64,S)
J = convert(Int64,J)
P = pclogit(ßest,Y,X,Z,J)

Ywtemp = reshape(Y,N,T,S)
@assert isequal(Ywtemp[:,:,1],Ywtemp[:,:,end])
Pw = zeros(N,T,S,J)
Yw = zeros(N,T,S,J)
for j=1:J
	Pw[:,:,:,j] = reshape(P[:,j],N,T,S,1)
	Yw[:,:,:,j] = Ywtemp.==j;
end

choice_like = ones(N,S)
for s=1:S
	choice_like[:,s] = squeeze(squeeze(prod(prod(Pw[:,:,s,:].^Yw[:,:,s,:],3),2),3),2)
end

lnWagew  = reshape(lnWage,N,T,S);
wageResw = reshape(Xwage*ω,N,T,S);
wage_like = ones(N,S);
for s=1:S
	wage_like[:,s] = prod(pdf(thisNormal,lnWagew[:,:,s]-wageResw[:,:,s]),2)
end

full_like = choice_like.*wage_like

return full_like
end
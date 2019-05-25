@views @inline function likecalc(Y,X,Z,lnWage,Xwage,βest,ωest,N::Int64,T::Int64,S::Int64,J::Int64)
    ω = ωest[1:end-1]
    σ = ωest[end]

    thisNormal = Normal(0,σ)

    P = pclogit(βest,Y,X,Z,J)

    Ywtemp = reshape(Y,N,T,S)
    @assert isequal(Ywtemp[:,:,1],Ywtemp[:,:,end])
    Pw = zeros(N,T,S,J)
    Yw = zeros(N,T,S,J)
    for j=1:J
        Pw[:,:,:,j] = reshape(P[:,j],N,T,S,1)
        Yw[:,:,:,j] = (Ywtemp.==j)
    end

    choice_like = ones(N,S)
    for s=1:S
        choice_like[:,s] = dropdims(dropdims(prod(prod(Pw[:,:,s,:].^Yw[:,:,s,:];dims=3);dims=2);dims=3);dims=2)
    end

    lnWagew  = reshape(lnWage,N,T,S);
    wageResw = reshape(Xwage*ω,N,T,S);
    wage_like = ones(N,S);
    for s=1:S
        wage_like[:,s] = prod(pdf.(thisNormal,lnWagew[:,:,s].-wageResw[:,:,s]);dims=2)
    end

    full_like = choice_like.*wage_like
end

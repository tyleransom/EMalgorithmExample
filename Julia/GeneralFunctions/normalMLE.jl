@views @inline function normalMLE(bstart::Vector,Y::Array,X::Array,W::Array=ones(length(Y)),d::Array=ones(length(Y)))

    J = length(unique(d))
    n = length(Y)
 
    # error checking
    @assert size(X,1)==n "The 1st dimension of X should equal the number of observations in Y"
    @assert minimum(d)==1 && maximum(d)==J  "d should contain integers numbered consecutively from 1 through J"
    @assert size(X,2)+J==size(bstart,1) "parameter vector has wrong number of elements"
    @assert size(W,1)==n "The size of the weighting vector should equal the number of observations in Y"
 
    function f(b)
        T         = promote_type(promote_type(eltype(X),eltype(b)),eltype(W))
        dmat      =  ones(T,n,J)
        likemat   = zeros(T,n,J)
        liker     = zeros(T,n)
        ℓ         =  zero(T)
        beta      = b[1:end-J]
        wagesigma = b[end-(J-1):end]
        for j=1:J
            dmat[:,j] .= d.==j
            likemat[:,j] .= vec( -0.5.*( log.(2 .* pi) .+ log.(wagesigma[j]^2) .+ ( (Y.-X*beta)./wagesigma[j] ).^2 ) )
        end
        liker .= vec(sum(dmat.*likemat;dims=2))
        ℓ = -W'*liker
    end

    function g!(G,b)
        T         = promote_type(promote_type(promote_type(eltype(X),eltype(b)),eltype(Y)),eltype(W))
        temp      = zeros(T,size(Y,1))
        beta      = b[1:end-J]
        wagesigma = b[end-(J-1):end]

        G .= T(0)
        for j=1:J
            G[1:end-J] .+= vec( -X'*( W.*(d.==j).*(Y.-X*beta)./(wagesigma[j].^2) ) )
        end
        for j=1:J
            k=length(b)-(J-1)+j-1
            temp .= vec( 1 ./ wagesigma[j] .- ( (Y.-X*beta).^2) ./ (wagesigma[j].^3 ) )
            G[k] = sum(W.*(d.==j).*temp)
        end
        return nothing
    end

    td = TwiceDifferentiable(f, g!, bstart, autodiff = :forwarddiff)
    rs = optimize(td, bstart, LBFGS(; linesearch = LineSearches.BackTracking()), Optim.Options(iterations=100_000,g_tol=1e-6,f_tol=1e-6))
    β  = Optim.minimizer(rs)
    ℓ  = Optim.minimum(rs)*(-1)
    H  = Optim.hessian!(td, β)
    g  = Optim.gradient!(td, β)
    se = sqrt.(diag(inv(H)))

    return β,se,ℓ,g
end

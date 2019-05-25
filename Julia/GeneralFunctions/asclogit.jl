@views @inline function asclogit(bstart::Vector,Y::Array,X::Array,Z::Array,J::Int64,baseAlt::Int64=J,W::Array=ones(length(Y)))

    ## error checking
    @assert ((!isempty(X) || !isempty(Z)) && !isempty(Y))    "You must supply data to the model"
    @assert (ndims(Y)==1 && size(Y,2)==1)                    "Y must be a 1-D Array"
    @assert (minimum(Y)==1 && maximum(Y)==J) "Y should contain integers numbered consecutively from 1 through J"
    if !isempty(X)
        @assert ndims(X)==2          "X must be a 2-dimensional matrix"
        @assert size(X,1)==size(Y,1) "The 1st dimension of X should equal the number of observations in Y"
    end
    if !isempty(Z)
        @assert ndims(Z)==3          "Z must be a 3-dimensional tensor"
        @assert size(Z,1)==size(Y,1) "The 1st dimension of Z should equal the number of observations in Y"
        @assert size(Z,3)==J         "The 3rd dimension of Z should equal the number of choice alternatives"
    end

    K1 = size(X,2)
    K2 = size(Z,2)
    jdx = setdiff(1:J,baseAlt)

    function f(b)
        T = promote_type(promote_type(promote_type(eltype(X),eltype(b)),eltype(Z)),eltype(W))
        num   = zeros(T,size(Y))
        dem   = zeros(T,size(Y))
        temp  = zeros(T,size(Y))
        numer =  ones(T,size(Y,1),J)
        P     = zeros(T,size(Y,1),J)
        ℓ     =  zero(T)
        b2 = b[K1*(J-1)+1:K1*(J-1)+K2]
                                             
        k = 1
        for j in 1:J
            if j != baseAlt
                temp       .= X*b[(k-1)*K1+1:k*K1] .+ (Z[:,:,j].-Z[:,:,baseAlt])*b2
                num        .= (Y.==j).*temp.+num
                dem        .+= exp.(temp)
                numer[:,j] .=  exp.(temp)
                k += 1
            end
        end
        dem.+=1
        P   .=numer./(1 .+ sum(numer;dims=2))

        ℓ = -W'*(num.-log.(dem))
    end

    function g!(G,b)
        T     = promote_type(promote_type(promote_type(eltype(X),eltype(b)),eltype(Z)),eltype(W))
        numer = zeros(T,size(Y,1),J)
        P     = zeros(T,size(Y,1),J)
        numg  = zeros(T,K2)
        demg  = zeros(T,K2)
        b2    = b[K1*(J-1)+1:K1*(J-1)+K2]
                                                                                                                                     
        G .= T(0)
        k = 1
        for j in 1:J
            if j != baseAlt
                numer[:,j] .= exp.( X*b[(k-1)*K1+1:k*K1] .+ (Z[:,:,j].-Z[:,:,baseAlt])*b2 )
                k += 1
            end
        end
        P   .=numer./(1 .+ sum(numer;dims=2))

        k = 1
        for j in 1:J
            if j != baseAlt
                G[(k-1)*K1+1:k*K1] .= -X'*(W.*((Y.==j).-P[:,j]))
                k += 1
            end
        end

        for j in 1:J
            if j != baseAlt
                numg .-= (Z[:,:,j].-Z[:,:,baseAlt])'*(W.*(Y.==j))
                demg .-= (Z[:,:,j].-Z[:,:,baseAlt])'*(W.*P[:,j])
            end
        end
        G[K1*(J-1)+1:K1*(J-1)+K2] .= numg.-demg
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

@views @inline function datagen(N::Int64=20_000,T::Int64=3)
    ## Generate data for a linear model to test optimization
    
    J = 5
    S = 2
    n = convert(Int64,N*T)
    
    ID = collect(1:N)*ones(1,T)
    Tw = ones(N,1)*collect(1:T)'
    
    piAns  = .3
    typer  = 2 .- (rand(N,1).>piAns)
    typerw = repeat(typer,1,T)
    
    # generate the covariates
    X = hcat(ones(n,1),5 .+ 3*randn(n,1),rand(n,1),2.5 .+ 2*randn(n,1),typerw[:].==1)
    K1 = size(X,2)
    Z = zeros(n,3,J)
    for j=1:J
        Z[:,:,j] .= hcat(3 .+ randn(n,1),randn(n,1).-1,rand(n,1))
    end
    K2 = size(Z,2)
    baseAlt = J

    # X coefficients
    βans      = zeros(size(X,2),J)
    βans[:,1] = [-0.15 0.10  0.50 0.10 -.15 ]
    βans[:,2] = [-1.50 0.15  0.70 0.20  .25 ]
    βans[:,3] = [-0.75 0.25 -0.40 0.30 -.05 ]
    βans[:,4] = [ 0.65 0.05 -0.30 0.40  .35 ]
    βans[:,5] = [ 0.75 0.10 -0.50 0.50 -.25 ]

    # Z coefficients
    γans = [.2 .5 .8]'

    # lnWage coefficients
    ωans = vcat(-0.15,0.10,0.50,0.10,-.15 )
    σans = .3

    # generate choice probabilities
    u   = zeros(n,J)
    p   = zeros(n,J)
    dem = zeros(n,1)
    for j=1:J
        u[:,j] .= vec(X*βans[:,j].+Z[:,:,j]*γans)
        dem.+=exp.(u[:,j])
    end
    for j=1:J
        p[:,j] .= vec(exp.(u[:,j])./dem)
    end
    
    # use the choice probabilities to create the observed choices
    draw=rand(n,1)
    Y=(draw.<sum(p[:,1:end];dims=2)).+
      (draw.<sum(p[:,2:end];dims=2)).+
      (draw.<sum(p[:,3:end];dims=2)).+
      (draw.<sum(p[:,4:end];dims=2)).+
      (draw.<sum(p[:,5:end];dims=2))
    Y=1.0.*Y
    
    β = broadcast(-,βans,βans[:,J])[1:end-K1]
    
    # generate wages
    Xwage  = X
    lnWage = Xwage*ωans+σans*randn(n,1)
    
    # return generated data so that other functions (below) have access
    return X,Y,Z,Xwage,lnWage,β,γans,ωans,σans,n,J,K1,K2,baseAlt,S,ID,Tw,typerw
end

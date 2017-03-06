function datagen(N,T)
    ## Generate data for a linear model to test optimization
    srand(1234)
    
    N = convert(Int64,N) #inputs to functions such as -ones- need to be integers!
    T = convert(Int64,T)
    J = 5
    S = 2
    n = convert(Int64,N*T) # use -const- as a way to declare a variable to be global (so other functions can access it)
    
    ID = collect(1:N)*ones(1,T)
    Tw = ones(N,1)*collect(1:T)'
    
    piAns  = .3
    typer  = 2-(rand(N,1).>piAns)
    typerw = repmat(typer,1,T)
    
    # generate the covariates
    X = [ones(n,1) 5+3*randn(n,1) rand(n,1) 2.5+2*randn(n,1) typerw[:].==1]
    K1 = size(X,2)
    Z = zeros(n,3,J)
    for j=1:J
        Z[:,:,j] = [3+randn(n,1) randn(n,1)-1 rand(n,1)]
    end
    K2 = size(Z,2)
    baseAlt = J

    # X coefficients
    ßans      = zeros(size(X,2),J)
    ßans[:,1] = [-0.15 0.10  0.50 0.10 -.15 ]
    ßans[:,2] = [-1.50 0.15  0.70 0.20  .25 ]
    ßans[:,3] = [-0.75 0.25 -0.40 0.30 -.05 ]
    ßans[:,4] = [ 0.65 0.05 -0.30 0.40  .35 ]
    ßans[:,5] = [ 0.75 0.10 -0.50 0.50 -.25 ]

    # Z coefficients
    γans = [.2 .5 .8]'

    # lnWage coefficients
    ωans = cat(1,-0.15,0.10,0.50,0.10,-.15 )
    σans = .3

    # generate choice probabilities
    u   = zeros(n,J)
    p   = zeros(n,J)
    dem = zeros(n,1)
    for j=1:J
        u[:,j] = X*ßans[:,j]+Z[:,:,j]*γans
        dem=exp(u[:,j])+dem
    end
    for j=1:J
        p[:,j] = exp(u[:,j])./dem
    end
    
    # use the choice probabilities to create the observed choices
    draw=rand(n,1)
    Y=(draw.<sum(p[:,1:end],2))+
      (draw.<sum(p[:,2:end],2))+
      (draw.<sum(p[:,3:end],2))+
      (draw.<sum(p[:,4:end],2))+
      (draw.<sum(p[:,5:end],2))
    
    ß = broadcast(-,ßans,ßans[:,J])[1:end-K1]
    
    # generate wages
    Xwage  = X
    lnWage = Xwage*ωans+σans*randn(n,1)
    
    # return generated data so that other functions (below) have access
    return X,Y,Z,Xwage,lnWage,ß,γans,ωans,σans,n,J,K1,K2,baseAlt,S,ID,Tw,typerw
end
# Simple estimation of finite mixture multinomial logit using simulated data
# Use EM algorithm to estimate type-specific coefficients
using Optim
using LineSearches
using Distributions
using Random
using LinearAlgebra

include("../GeneralFunctions/typeprob.jl")
include("../GeneralFunctions/normalMLE.jl")
include("../GeneralFunctions/pclogit.jl")
include("../GeneralFunctions/asclogit.jl")
include("likecalc.jl")
include("datagen.jl")

Random.seed!(1234)

function datersim()
    # Simulate data
    N = 1_000
    T = 5
    @time X,Y,Z,Xwage,lnWage,βans,γans,ωans,σans,n,J,K1,K2,baseAlt,S,IDw,Tw,typerw = datagen(N,T)

    # Now estimate without knowing unobserved types:
    # Treat last element of X as unobserved, and double the data
    Xfeas      = hcat(kron(ones(S,1),X[:,1:end-1]),kron(Matrix(1.0I,S,S-1),ones(size(X,1))))
    Xwagefeas  = Xfeas
    lnWagefeas = repeat(lnWage,S,1)
    Yfeas      = repeat(Y,S,1)
    IDfeas     = repeat(IDw[:],S,1)
    Twfeas     = repeat(Tw[:],S,1)
    Zfeas      = repeat(Z,S,1,1)

    # EM algorithm starting values
    prior = hcat(.55,.45)
    βest = vcat(βans,γans) .+ .5 .*rand(Float64,size(vcat(βans,γans))) .* vcat(βans,γans) .- .25 .* vcat(βans,γans)
    ωest = vcat(ωans,σans) .+ .5 .*rand(Float64,size(vcat(ωans,σans))) .* vcat(ωans,σans) .- .25 .* vcat(ωans,σans)
    EMcrit = 1
    iteration = 1

    full_like = likecalc(Yfeas,Xfeas,Zfeas,lnWagefeas,Xwagefeas,βest,ωest,N,T,S,J)
    prior,Ptype,Ptypel,jointlike = typeprob(prior,full_like,T)
    println("")
    println("Initial likelihood value = ",jointlike)

    while EMcrit>1e-4
    #while iteration<10
        oPtype = Ptype
        # E step
        full_like = @time likecalc(Yfeas,Xfeas,Zfeas,lnWagefeas,Xwagefeas,βest,ωest,N,T,S,J)
        prior,Ptype,Ptypel,jointlike = @time typeprob(prior,full_like,T)
        # M step
        βest,se_βest,ℓ,g = @time asclogit(vec(βest),vec(Yfeas),Xfeas,Zfeas,size(Zfeas,3),size(Zfeas,3),vec(Ptypel))
        P = @time pclogit(βest,Y,X,Z,J)
        ωest,se_ωest,ℓ,g = @time normalMLE(vec(ωest),lnWagefeas,Xwagefeas,vec(Ptypel))
        EMcrit = norm(Ptype[:].-oPtype[:],Inf)
        iteration += 1
        println("")
        println("Likelihood value = ",jointlike)
        println("Pr(type==1) is ",prior[1])
        println("Iteration is ",iteration)
        println("EM criterion is ",EMcrit)
        println("")
    end

    # compare answers
    compareAns=hcat(βest,vcat(βans,γans))
    println(compareAns[:,1])
    println(compareAns[:,2])
    compareAns=hcat(ωest,vcat(ωans,σans))
    println(compareAns[:,1])
    println(compareAns[:,2])
end
datersim()

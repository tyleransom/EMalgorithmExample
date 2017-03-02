# Simple estimation of finite mixture multinomial logit using simulated data
using JuMP
using Ipopt

include("../GeneralFunctions/typeprob.jl")
include("../GeneralFunctions/normalMLE.jl")
include("../GeneralFunctions/pclogit.jl")
include("../GeneralFunctions/clogit.jl")
include("datagen.jl")

# Simulate data
@time X,Y,Z,ßans,γans,n,J,K1,K2,baseAlt,S = datagen(1e4,3)

# optimize
startval = rand(size(cat(1,ßans[:],γans)));
@time ßopt,γopt = clogit(startval,Y,X,Z,size(Z,3),ones(size(Y)))

# compare answers
compareAns=cat(2,cat(1,ßans[:],γans),cat(1,ßopt,γopt))
println(compareAns[:,1])
println(compareAns[:,2])

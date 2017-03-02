# Simple estimation of finite mixture multinomial logit using simulated data
# Use EM algorithm to estimate type-specific coefficients
using JuMP
using Ipopt

include("../GeneralFunctions/typeprob.jl")
include("../GeneralFunctions/normalMLE.jl")
include("../GeneralFunctions/pclogit.jl")
include("../GeneralFunctions/clogit.jl")
include("likecalc.jl")
include("datagen.jl")

# Simulate data
N = 1e3
T = 3
@time X,Y,Z,ßans,γans,n,J,K1,K2,baseAlt,S,IDw,Tw,typerw = datagen(N,T)

# Now estimate without knowing unobserved types:
# Treat last element of X as unobserved, and double the data
Xfeas  = cat(2,kron(ones(S,1),X[:,1:end-1]),kron(eye(S,S-1),ones(size(X,1),1)))
Yfeas  = kron(ones(S,1),Y)
IDfeas = kron(ones(S,1),IDw[:])
Twfeas = kron(ones(S,1),Tw[:])
Zfeas  = Z
for s=1:S-1
    Zfeas = cat(1,Zfeas,Z)
end

# EM algorithm starting values
prior = cat(2,.65,.35)
ßest = cat(1,ßans,γans)
# ßest[size(X,2):size(X,2):(J-1)*size(X,2)] = rand(size(ßest[size(X,2):size(X,2):(J-1)*size(X,2)])) 
EMcrit = 1
iteration = 1

full_like = likecalc(Yfeas,Xfeas,Zfeas,ßest,N,T,S,J)
prior,Ptype,Ptypel,jointlike = typeprob(prior,full_like,T)
println("")
println("Initial likelihood value = ",jointlike)

typebs = ßest[size(X,2):size(X,2):(J-1)*size(X,2)]

while EMcrit>1e-5
    oPtype = Ptype
    # E step
    full_like = likecalc(Yfeas,Xfeas,Zfeas,ßest,N,T,S,J)
    prior,Ptype,Ptypel,jointlike = typeprob(prior,full_like,T)
    # M step
    ßopt,γopt = clogit(startval,Yfeas,Xfeas,Zfeas,size(Zfeas,3),Ptypel)
    ßest = cat(1,ßopt,γopt)
    typebs=cat(2,typebs,ßest[size(X,2):size(X,2):(J-1)*size(X,2)])
    EMcrit = norm(Ptype[:]-oPtype[:],Inf)
    iteration += 1
    println("")
    println("Likelihood value = ",jointlike)
    println("Pr(type==1) is ",prior[1])
    println("Iteration is ",iteration)
    println("EM criterion is ",EMcrit)
    println("")
    # save tempResults *feas typebs iteration
end

# # # # compare answers
# # # compareAns=cat(2,cat(1,ßans[:],γans),cat(1,ßopt,γopt))
# # # println(compareAns[:,1])
# # # println(compareAns[:,2])

# Import packages
using DelimitedFiles
using LinearAlgebra: diag
using Optim
using LineSearches

include("../GeneralFunctions/normalMLE.jl")

# Read in auto data from Stata example
dat  = readdlm("/home/data/ransom/learningCSUF/Simulations/Julia_asclogit/auto.csv",','; skipstart=1);
data = convert(Array{Float64},dat[vec(dat[:,4].!=""),2:end-2])
price = data[:,1]
mpg = data[:,2]
rep78 = data[:,3]
headroom = data[:,4]
trunk = data[:,5]
X = hcat(ones(length(price)),mpg,headroom,trunk)
Y = log.(price)

θ  = [9.18364, -.0271639,  -.1122285, .026621,  .33626]

println(size(.1*ones(size(X,2)+1)))
println(size(Y))
println(size(X))
println(size(rep78))
println(typeof(.1*ones(size(X,2)+1)))
println(typeof(Y))
println(typeof(X))
println(typeof(rep78))
β,se_β,like,grad = normalMLE(.1*ones(size(X,2)+1),Y,X)
β,se_β,like,grad = @time normalMLE(.1*ones(size(X,2)+1),Y,X)
println([β se_β])
println(like)

β,se_β,like,grad = @time normalMLE(.1*ones(size(X,2)+1),Y,X,rep78)
println([β se_β])
println(like)

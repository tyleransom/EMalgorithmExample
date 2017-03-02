function clogit(startval,Y,X,Z,baseAlt,W)
    ## MLE of multinomial logit model
    # Declare the name of your model and the optimizer you will use
    clogitMLE = Model(solver=IpoptSolver(tol=1e-6,print_level=0))
    
    # Check that inputs make sense
    @assert size(Y,2)==1
    @assert size(Y,1)==size(X,1)
    @assert size(Y,1)==size(Z,1)
    @assert size(Y,1)==size(Z,1)
    
    K1 = size(X,2)
    K2 = size(Z,2)
    J = size(Z,3)
    Zdiff = broadcast(-,Z,Z[:,:,J])
    
    # Declare the variables you are optimizing over
    @variable(clogitMLE, ß[k1=1:K1*(J-1)], start=startval[k1])
    @variable(clogitMLE, γ[k2=1:K2]                          )
    
    # # Write constraints here if desired (in this case, set betas for base alternative to be 0)
    # for k=1:K1
      # @constraint(clogitMLE, ß[k,baseAlt] == 0)
    # end
    
    # Write the objective function to be maximized
    @NLobjective(clogitMLE, Max, sum( W[i]*(sum( (Y[i]==j)*( sum( X[i,k1]*ß[(j-1)*K1+k1] for k1=1:K1) + 
                                 sum( Zdiff[i,k2,j]*γ[k2] for k2=1:K2) ) for j=1:J-1)
                               - log( 1 + sum( exp( sum( X[i,k1]*ß[(j-1)*K1+k1] for k1=1:K1) + 
                                 sum( Zdiff[i,k2,j]*γ[k2] for k2=1:K2)  ) for j=1:J-1) )) for i=1:n))
    
    # Solve the objective function
    status = solve(clogitMLE)
    
    # # Generate Hessian
    # this_par = clogitMLE.colVal
    # m_eval = JuMP.NLPEvaluator(clogitMLE);
    # MathProgBase.initialize(m_eval, [:ExprGraph, :Grad, :Hess])
    # hess_struct = MathProgBase.hesslag_structure(m_eval)
    # hess_vec = zeros(length(hess_struct[1]))
    # numconstr = length(m_eval.m.linconstr) + length(m_eval.m.quadconstr) + length(m_eval.m.nlpdata.nlconstr)
    # dimension = length(clogitMLE.colVal)
    # MathProgBase.eval_hesslag(m_eval, hess_vec, this_par, 1.0, zeros(numconstr))
    # this_hess_ld = sparse(hess_struct[1], hess_struct[2], hess_vec, dimension, dimension)
    # hopt = this_hess_ld + this_hess_ld' - sparse(diagm(diag(this_hess_ld)));
    # hopt = -full(hopt); #since we are maximizing
    
    # # Calculate standard errors
    # seOpt = sqrt(diag(hOpt\eye(size(hOpt,1))))
    # # swap indices because order of coefficients is different than ordering of Hessian rows
    # # note: it would be better to do this re-ordering to the Hessian, but I can't figure out...
    # # how to do this simultaneously for rows and columns
    # ind = [1:size(hOpt,1)]
    # for j=1:J-1
        # ind[(j-1)*K1+1:j*K1] = [j:K1:(J-1)*K1]
    # end
    # seOpt = seOpt[ind]
    
    # Save estimates
    ßopt = getvalue(ß[:])
    γopt = getvalue(γ[:])
    # hopt  = -eye(length(ß)+length(γ))
    
    # Print estimates and log likelihood value
    # println("beta = ", ßopt[:])
    # println("γ = ", γopt[:])
    # println("MLE objective: ", getobjectivevalue(clogitMLE))
    # println("MLE status: ", status)
    return ßopt,γopt
end
function jumpMLE(startval,y,X,W)
    ## MLE of classical linear regression model
    # Declare the name of your model and the optimizer you will use
    myMLE = Model(solver=IpoptSolver(tol=1e-6))
    
    # Declare the variables you are optimizing over
    @variable(myMLE, ß[i=1:size(X,2)], start = startval[i])
    @variable(myMLE, σ>=0.0, start = startval[end])
    # @variable(myMLE, σ ==0.5) # use this syntax if you want to restrict a parameter to a specified value. Note that NLopt has issues with equality constraints
    
    # Write constraints here if desired
    
    # Write your objective function
    @NLobjective(myMLE, Max, sum( W[i]*(.5*log(1/(2*π*σ^2))-(y[i]-sum(X[i,k]*ß[k] for k=1:size(X,2)))^2/(2σ^2)) for i=1:size(X,1)))
    # Solve the objective function
    status = solve(myMLE)
    
    # # Generate Hessian
    # this_par = myMLE.colVal
    # m_eval = JuMP.NLPEvaluator(myMLE);
    # MathProgBase.initialize(m_eval, [:ExprGraph, :Grad, :Hess])
    # hess_struct = MathProgBase.hesslag_structure(m_eval)
    # hess_vec = zeros(length(hess_struct[1]))
    # numconstr = length(m_eval.m.linconstr) + length(m_eval.m.quadconstr) + length(m_eval.m.nlpdata.nlconstr)
    # dimension = length(myMLE.colVal)
    # MathProgBase.eval_hesslag(m_eval, hess_vec, this_par, 1.0, zeros(numconstr))
    # this_hess_ld = sparse(hess_struct[1], hess_struct[2], hess_vec, dimension, dimension)
    # hOpt = this_hess_ld + this_hess_ld' - sparse(diagm(diag(this_hess_ld)));
    # hOpt = -full(hOpt); #since we are maximizing
    
    # # Calculate standard errors
    # seOpt = sqrt(diag(full(hOpt)\eye(size(hOpt,1))))
    
    # Save estimates
    ßopt = getvalue(ß[:]);
    σopt = getvalue(σ);
    
    # Print estimates and log likelihood value
    # println("beta = ", bOpt)
    # println("s = ", sOpt)
    println("MLE objective: ", getobjectivevalue(myMLE))
    println("MLE status: ", status)
    return ßopt,σopt
end
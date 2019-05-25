@views @inline function pclogit(b,Y,X,Z,baseAlt=maximum(Y))
     
    # error checking
    @assert (~isempty(X) || ~isempty(Z)) && ~isempty(Y)
     
    N  = size(Y,1)
    K1 = size(X,2)
    K2 = size(Z,2)
    J  = size(Z,3)
     
    @assert length(b)==(K1*(J-1)+K2)
    @assert ndims(Y)==2 && size(Y,2)==1
    if ~isempty(X)
        @assert ndims(X)==2
        @assert size(X,1)==N
    end
    if ~isempty(Z)
        @assert ndims(Z)==3
        @assert size(Z,1)==N
        @assert size(Z,3)==J
    end
     
    b2   = b[K1*(J-1)+1:K1*(J-1)+K2]
    num  =  ones(N,J)
    temp = zeros(N)
     
    if K2>0 && K1>0
        # sets BASEALT to be the alternative that is normalized to zero
        k = 1
        for j=setdiff(1:J,baseAlt)
            temp.=X*b[(k-1)*K1+1:k*K1].+(Z[:,:,j].-Z[:,:,baseAlt])*b2
            num[:,j].=exp.(temp)
            k+=1
        end
    elseif K1>0 && K2==0
        # sets BASEALT to be the alternative that is normalized to zero
        k = 1
        for j=setdiff(1:J,baseAlt)
            temp.=X*b[(k-1)*K1+1:k*K1]
            num[:,j].=exp.(temp)
            k+=1
        end
    elseif K1==0 && K2>0
        # sets BASEALT to be the alternative that is normalized to zero
        k = 1
        for j=setdiff(1:J,baseAlt)
            temp.=(Z[:,:,j].-Z[:,:,baseAlt])*b2
            num[:,j].=exp.(temp)
            k+=1
        end
    end
     
    P=num./sum(num;dims=2)
end
